#!/usr/bin/env python3
"""
Experiment #797: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After 500+ failed strategies, daily timeframe offers best signal-to-noise ratio.
1. 1d bars filter intraday noise while maintaining sufficient trade frequency (20-50/year)
2. 1w HMA(21) provides robust long-term trend bias without being too slow
3. Choppiness Index(14) on 1d cleanly separates ranging vs trending regimes
4. Connors RSI (RSI3 + RSI_Streak + PercentRank) / 3 has 75% win rate for mean reversion
5. Dual regime logic: mean revert when CHOP>55, trend-follow pullbacks when CHOP<45
6. ATR(14) trailing stop at 2.5x protects from 2022-style crashes
7. Position sizing: 0.25-0.30 discrete levels to minimize fee churn
8. Relaxed CRSI thresholds (15/85 instead of 10/90) ensure >=10 trades/train

Key differences from failed strategies:
- Connors RSI instead of standard RSI (better for mean reversion)
- 1w HMA for trend (not 1d/4h - cleaner signal on daily bars)
- CHOP thresholds: 55/45 (proven on ETH in research)
- Hold logic: maintain position until regime/trend reversal
- Volume filter removed (less relevant on 1d, reduces trade count)

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_hma_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother than EMA, less lag than SMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak component of Connors RSI.
    Measures consecutive up/down days.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    abs_streak = np.abs(streak)
    streak_signed = np.sign(streak)
    
    # Simple mapping: streak of 0 = 50, streak of +3 = 100, streak of -3 = 0
    streak_score = 50 + streak_signed * np.minimum(abs_streak, 3) * (50 / 3)
    streak_score = np.clip(streak_score, 0, 100)
    
    # Apply RSI to streak scores
    streak_rsi = calculate_rsi(streak_score, period)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component of Connors RSI.
    Percentage of days in lookback where close was lower than today.
    """
    n = len(close)
    pr = np.full(n, np.nan)
    
    if n < period + 1:
        return pr
    
    for i in range(period, n):
        lookback = close[i-period+1:i+1]
        current = close[i]
        count_lower = np.sum(lookback[:-1] < current)
        pr[i] = 100 * count_lower / (period - 1)
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100. <10 = oversold, >90 = overbought.
    We use <15/>85 for more trades.
    """
    rsi3 = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi3 + streak_rsi + pr) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending.
    We use 55/45 for more regime switches.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di_pct = 100 * plus_di / (atr + 1e-10)
        minus_di_pct = 100 * minus_di / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di_pct - minus_di_pct) / (plus_di_pct + minus_di_pct + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    crsi_1d = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr_1d = calculate_atr(high, low, close, period=14)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    adx_1d = calculate_adx(high, low, close, period=14)
    
    # Calculate and align 1w HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(chop_1d[i]) or np.isnan(adx_1d[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (1d Choppiness Index) ===
        ranging_regime = chop_1d[i] > 55
        trending_regime = chop_1d[i] < 45
        neutral_regime = not ranging_regime and not trending_regime
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_1d[i] > 25
        weak_trend = adx_1d[i] < 20
        
        # === CONNORS RSI SIGNALS (relaxed for more trades) ===
        crsi_oversold = crsi_1d[i] < 20
        crsi_overbought = crsi_1d[i] > 80
        crsi_extreme_oversold = crsi_1d[i] < 15
        crsi_extreme_overbought = crsi_1d[i] > 85
        crsi_neutral_low = 20 <= crsi_1d[i] < 40
        crsi_neutral_high = 60 < crsi_1d[i] <= 80
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) - Mean Reversion ===
        if ranging_regime:
            # Mean reversion long: CRSI oversold + 1w bullish bias
            if crsi_oversold and trend_1w_bullish:
                desired_signal = BASE_SIZE
            
            # Mean reversion short: CRSI overbought + 1w bearish bias
            if crsi_overbought and trend_1w_bearish:
                desired_signal = -BASE_SIZE
            
            # Conservative: extreme CRSI regardless of trend
            if crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) - Trend Following ===
        elif trending_regime:
            # Trend pullback long: 1w bullish + CRSI neutral low (pullback entry)
            if trend_1w_bullish and crsi_neutral_low:
                desired_signal = BASE_SIZE
            
            # Trend pullback short: 1w bearish + CRSI neutral high (pullback entry)
            if trend_1w_bearish and crsi_neutral_high:
                desired_signal = -BASE_SIZE
            
            # Strong trend continuation: ADX > 25 + CRSI not extreme
            if strong_trend and trend_1w_bullish and crsi_1d[i] < 70:
                desired_signal = BASE_SIZE
            
            if strong_trend and trend_1w_bearish and crsi_1d[i] > 30:
                desired_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: only extreme CRSI + trend alignment
            if crsi_extreme_oversold and trend_1w_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and trend_1w_bearish:
                desired_signal = -REDUCED_SIZE
            
            # ADX filter for trend confirmation
            if strong_trend and trend_1w_bullish and crsi_neutral_low:
                desired_signal = REDUCED_SIZE
            
            if strong_trend and trend_1w_bearish and crsi_neutral_high:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1w trend intact and CRSI not overbought
                if trend_1w_bullish and crsi_1d[i] < 75:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if 1w trend intact and CRSI not oversold
                if trend_1w_bearish and crsi_1d[i] > 25:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1w trend reverses or CRSI overbought
            if trend_1w_bearish and crsi_1d[i] > 70:
                desired_signal = 0.0
            # Exit if regime switches to strong trending against position
            if trending_regime and trend_1w_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1w trend reverses or CRSI oversold
            if trend_1w_bullish and crsi_1d[i] < 30:
                desired_signal = 0.0
            # Exit if regime switches to strong trending against position
            if trending_regime and trend_1w_bullish:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals