#!/usr/bin/env python3
"""
Experiment #803: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. 1d timeframe reduces noise and fee drag vs lower TFs (target 20-40 trades/year)
2. 1w HMA(21) provides robust long-term trend bias without whipsaw
3. Connors RSI (RSI3 + RSI_Streak2 + PercentRank100) / 3 has 75% win rate in research
4. Choppiness Index(14) regime filter switches between mean-revert and trend-follow
5. Asymmetric sizing: smaller positions in bear markets (1w HMA bearish)
6. ATR(14) trailing stop at 2.5x protects from major drawdowns
7. Relaxed CRSI thresholds (15/85 instead of 10/90) ensures sufficient trades
8. Volume confirmation filter reduces false breakouts

Strategy design:
1. 1w HMA(21) for long-term trend bias (aligned via mtf_data helper)
2. 1d Choppiness Index(14) for regime detection (ranging vs trending)
3. 1d Connors RSI for entry timing (proven mean-reversion indicator)
4. 1d ATR(14) for trailing stop (2.5x)
5. 1d Bollinger Bands(20, 2.0) for additional mean-reversion confirmation
6. Discrete signals: 0.0, ±0.25, ±0.30
7. Asymmetric sizing: 0.25 in bear, 0.30 in bull

Key differences from failed 1d strategies:
- Connors RSI instead of standard RSI (faster reaction, better win rate)
- 1w HMA for trend (not 1d - avoids whipsaw on daily)
- CRSI thresholds: 15/85 (not 10/90) — generates more trades
- CHOP thresholds: 55/45 — more regime switches
- Hold logic: maintain position until opposite signal or stoploss
- Volume filter: 1.2x (not 1.5x) — less restrictive

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_hma_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - reduces lag while maintaining smoothness."""
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
    RSI of Streak - measures consecutive up/down days.
    Streak: +1 for up day, -1 for down day, cumulative.
    Then calculate RSI on absolute streak values.
    """
    n = len(close)
    rsi_streak = np.full(n, np.nan)
    
    if n < period + 2:
        return rsi_streak
    
    # Calculate streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = max(0, streak[i-1]) + 1
        elif close[i] < close[i-1]:
            streak[i] = min(0, streak[i-1]) - 1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on streak (using absolute values for up/down)
    delta = np.diff(streak)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + rs))
    
    rsi_streak = np.clip(rsi_streak, 0, 100)
    return rsi_streak

def calculate_percent_rank(close, period=100):
    """
    Percent Rank of daily returns over lookback period.
    Measures where current return ranks vs past N days.
    """
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    if n < period + 1:
        return pct_rank
    
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0], returns])
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / period
        pct_rank[i] = rank * 100
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100. <10 = extremely oversold, >90 = extremely overbought.
    We use 15/85 for more trade signals.
    """
    rsi = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, rank_period)
    
    crsi = (rsi + rsi_streak + pct_rank) / 3.0
    return crsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

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
    We use 55/45 for more regime switches on 1d.
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

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_1d = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_mult=2.0)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    vol_sma_1d = calculate_volume_sma(volume, period=20)
    
    # Calculate and align 1w HMA for long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE_BULL = 0.30  # Larger size in bull market
    BASE_SIZE_BEAR = 0.25  # Smaller size in bear market
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(bb_sma[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(chop_1d[i]):
            continue
        if np.isnan(vol_sma_1d[i]) or vol_sma_1d[i] <= 1e-10:
            continue
        
        # === LONG-TERM TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # Select position size based on trend
        base_size = BASE_SIZE_BULL if trend_1w_bullish else BASE_SIZE_BEAR
        
        # === REGIME DETECTION (1d Choppiness Index) ===
        ranging_regime = chop_1d[i] > 55
        trending_regime = chop_1d[i] < 45
        neutral_regime = not ranging_regime and not trending_regime
        
        # === VOLUME CONFIRMATION (relaxed) ===
        volume_confirmed = volume[i] > 1.2 * vol_sma_1d[i]
        
        # === CONNORS RSI SIGNALS (relaxed thresholds for more trades) ===
        crsi_extreme_oversold = crsi_1d[i] < 15
        crsi_extreme_overbought = crsi_1d[i] > 85
        crsi_oversold = crsi_1d[i] < 25
        crsi_overbought = crsi_1d[i] > 75
        crsi_neutral_low = 25 <= crsi_1d[i] <= 45
        crsi_neutral_high = 55 <= crsi_1d[i] <= 75
        
        # === BOLLINGER POSITION ===
        below_bb_lower = close[i] < bb_lower[i]
        above_bb_upper = close[i] > bb_upper[i]
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Mean reversion long: CRSI oversold + below BB lower
            if crsi_oversold and below_bb_lower:
                desired_signal = base_size if volume_confirmed else REDUCED_SIZE
            
            # Mean reversion short: CRSI overbought + above BB upper
            if crsi_overbought and above_bb_upper:
                desired_signal = -base_size if volume_confirmed else -REDUCED_SIZE
            
            # Conservative: extreme CRSI alone
            if crsi_extreme_oversold and not in_position:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and not in_position:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Trend pullback long: 1w bullish + CRSI neutral low (pullback entry)
            if trend_1w_bullish and crsi_neutral_low:
                desired_signal = base_size if volume_confirmed else REDUCED_SIZE
            
            # Trend pullback short: 1w bearish + CRSI neutral high (pullback entry)
            if trend_1w_bearish and crsi_neutral_high:
                desired_signal = -base_size if volume_confirmed else -REDUCED_SIZE
            
            # Breakout continuation with volume
            if trend_1w_bullish and above_bb_upper and volume_confirmed and crsi_1d[i] < 70:
                desired_signal = base_size
            
            if trend_1w_bearish and below_bb_lower and volume_confirmed and crsi_1d[i] > 30:
                desired_signal = -base_size
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) — Conservative ===
        else:
            # Only extreme CRSI entries in neutral regime
            if crsi_extreme_oversold and trend_1w_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and trend_1w_bearish:
                desired_signal = -REDUCED_SIZE
            
            # Allow basic mean reversion with volume confirmation
            if crsi_oversold and below_bb_lower and trend_1w_bullish:
                desired_signal = REDUCED_SIZE if volume_confirmed else 0.0
            
            if crsi_overbought and above_bb_upper and trend_1w_bearish:
                desired_signal = -REDUCED_SIZE if volume_confirmed else 0.0
        
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
                if trend_1w_bullish and crsi_1d[i] < 80:
                    desired_signal = base_size
            elif position_side < 0:
                # Hold short if 1w trend intact and CRSI not oversold
                if trend_1w_bearish and crsi_1d[i] > 20:
                    desired_signal = -base_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1w trend reverses or CRSI extremely overbought
            if trend_1w_bearish and crsi_1d[i] > 85:
                desired_signal = 0.0
            # Exit if price hits BB upper in ranging regime
            if ranging_regime and above_bb_upper and crsi_1d[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1w trend reverses or CRSI extremely oversold
            if trend_1w_bullish and crsi_1d[i] < 15:
                desired_signal = 0.0
            # Exit if price hits BB lower in ranging regime
            if ranging_regime and below_bb_lower and crsi_1d[i] < 30:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= base_size:
                desired_signal = base_size
            else:
                desired_signal = REDUCED_SIZE
        elif desired_signal < 0:
            if desired_signal <= -base_size:
                desired_signal = -base_size
            else:
                desired_signal = -REDUCED_SIZE
        
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