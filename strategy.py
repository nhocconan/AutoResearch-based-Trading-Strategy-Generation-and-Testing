#!/usr/bin/env python3
"""
Experiment #773: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + Donchian Breakout

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612 on 4h):
1. 1d timeframe has been failing recently (#763, #767, #771, #772 all negative Sharpe)
2. Choppiness Index regime detection showed ETH Sharpe +0.923 in research
3. Connors RSI mean reversion works best in ranging markets (CHOP > 61.8)
4. Donchian breakouts work best in trending markets (CHOP < 38.2)
5. 1w HMA(21) provides stronger trend bias than 1d for daily entries
6. Need to ensure adequate trade frequency (loosen entry thresholds vs failed 1d attempts)
7. ALL symbols must have Sharpe > 0 — need logic that works on BTC/ETH/SOL individually

Strategy design:
1. 1w HMA(21) for primary trend bias (aligned via mtf_data helper)
2. 1d Choppiness Index(14) for regime detection (range vs trend)
3. 1d Connors RSI for mean reversion entries in ranging regime
4. 1d Donchian(20) breakout for trend entries in trending regime
5. 1d ATR(14) for trailing stop (2.5x) and volatility normalization
6. Discrete signals: 0.0, ±0.25, ±0.30 (max 0.35)
7. Position sizing: 0.25-0.30 (conservative for drawdown control)

Key improvements from failed 1d attempts:
- Looser CRSI thresholds (15/85 vs 10/90) for more trades
- Choppiness as regime switch, not just filter
- 1w HMA instead of 1d for stronger trend signal
- Ensure trade frequency: target 20-40 trades/year on 1d
- Simpler hold logic to reduce overfitting

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
Timeframe: 1d (target 20-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_crsi_donchian_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average for trend detection."""
    series = pd.Series(series)
    wma1 = series.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = series.ewm(span=period, min_periods=period, adjust=False).mean()
    hull = 2 * wma1 - wma2
    hma = hull.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_ema(series, period):
    """Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

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
    Connors RSI Streak component.
    Measures consecutive up/down days.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    # Calculate streak values
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (inverted for down streaks)
    streak_abs = np.abs(streak)
    streak_score = np.where(streak >= 0, streak_abs, -streak_abs)
    
    # Calculate RSI of streak scores
    streak_rsi = calculate_rsi(streak_score + 100, period)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Connors RSI Percent Rank component.
    Percentage of past returns less than current return.
    """
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    if n < period + 1:
        return pct_rank
    
    returns = np.diff(close) / (close[:-1] + 1e-10) * 100
    returns = np.concatenate([[0], returns])
    
    for i in range(period, n):
        window = returns[i-period:i]
        current = returns[i]
        pct_rank[i] = 100 * np.sum(window < current) / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100. <15 = oversold, >85 = overbought.
    """
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pct_rank_period)
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi_short + streak_rsi + pct_rank) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    Measures market choppiness vs trending.
    CHOP > 61.8 = ranging/choppy
    CHOP < 38.2 = trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Calculate highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channels for breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF HMA for trend bias
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
    
    # Track previous regime for hysteresis
    prev_regime = 0  # 0=neutral, 1=trending, 2=ranging
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]) or np.isnan(chop_1d[i]) or np.isnan(atr_1d[i]):
            continue
        if atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
        # Use hysteresis to reduce flip-flop
        if chop_1d[i] > 58:
            regime = 2  # ranging
        elif chop_1d[i] < 42:
            regime = 1  # trending
        else:
            regime = prev_regime  # maintain previous
        
        prev_regime = regime
        trending_regime = (regime == 1)
        ranging_regime = (regime == 2)
        
        # === CRSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi_1d[i] < 18
        crsi_overbought = crsi_1d[i] > 82
        crsi_extreme_oversold = crsi_1d[i] < 12
        crsi_extreme_overbought = crsi_1d[i] > 88
        
        # === DONCHIAN BREAKOUT (Trend Following) ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 58) ===
        if ranging_regime:
            # Mean reversion long: CRSI oversold + trend not bearish
            if crsi_oversold and not trend_1w_bearish:
                desired_signal = BASE_SIZE
            
            # Mean reversion short: CRSI overbought + trend not bullish
            if crsi_overbought and not trend_1w_bullish:
                desired_signal = -BASE_SIZE
            
            # Conservative: extreme CRSI regardless of trend
            if crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 42) ===
        elif trending_regime:
            # Trend breakout long: Donchian breakout + 1w bullish
            if breakout_long and trend_1w_bullish:
                desired_signal = BASE_SIZE
            
            # Trend breakout short: Donchian breakout + 1w bearish
            if breakout_short and trend_1w_bearish:
                desired_signal = -BASE_SIZE
            
            # Trend pullback long: 1w bullish + CRSI neutral
            if trend_1w_bullish and 30 < crsi_1d[i] < 50:
                desired_signal = REDUCED_SIZE
            
            # Trend pullback short: 1w bearish + CRSI neutral
            if trend_1w_bearish and 50 < crsi_1d[i] < 70:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (42 <= CHOP <= 58) ===
        else:
            # Conservative: only extreme CRSI
            if crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            
            # Or trend breakout with confirmation
            if breakout_long and trend_1w_bullish:
                desired_signal = REDUCED_SIZE
            
            if breakout_short and trend_1w_bearish:
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
                # Hold long if trend intact and CRSI not overbought
                if trend_1w_bullish and crsi_1d[i] < 80:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if trend_1w_bearish and crsi_1d[i] > 20:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or CRSI overbought
            if trend_1w_bearish and crsi_1d[i] > 70:
                desired_signal = 0.0
            # Exit if CRSI reaches overbought in ranging regime
            if ranging_regime and crsi_1d[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or CRSI oversold
            if trend_1w_bullish and crsi_1d[i] < 30:
                desired_signal = 0.0
            # Exit if CRSI reaches oversold in ranging regime
            if ranging_regime and crsi_1d[i] < 25:
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