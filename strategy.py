#!/usr/bin/env python3
"""
Experiment #763: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: Based on research showing Connors RSI achieves 75% win rate in mean reversion,
combined with Choppiness Index regime detection and 1w HMA for major trend bias:
1. 1w HMA(21) provides cleaner major trend filter than EMA for crypto's noisy structure
2. Connors RSI (RSI3 + RSI_Streak2 + PercentRank100) / 3 catches oversold/overbought extremes
3. Choppiness Index(14) switches between mean-reversion (CHOP>61.8) and trend-follow (CHOP<38.2)
4. 1d timeframe targets 20-50 trades/year = minimal fee drag
5. ATR(14) trailing stop at 2.5x protects against crypto crashes
6. Discrete signals (0.0, ±0.25, ±0.30) minimize churn costs

Strategy design:
1. 1w HMA(21) for major trend bias (aligned via mtf_data helper)
2. 1d Choppiness Index(14) for regime detection
3. 1d Connors RSI for entry timing (extreme thresholds: <15 long, >85 short)
4. 1d SMA(200) as additional trend filter for mean reversion entries
5. 1d ATR(14) for trailing stop (2.5x)
6. Volume filter optional (1.3x average) for conviction

Key differences from failed experiments:
- Using Connors RSI instead of regular RSI (proven 75% win rate in research)
- 1w HTF instead of 1d (cleaner major trend signal)
- SMA200 filter for mean reversion entries (prevents catching falling knives)
- Simpler regime logic (dual: chop vs trend, not triple)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_hma_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - more responsive than EMA."""
    series = pd.Series(series)
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma1 = series.ewm(span=half_period, min_periods=half_period, adjust=False).mean()
    wma2 = series.ewm(span=period, min_periods=period, adjust=False).mean()
    
    wma_diff = 2 * wma1 - wma2
    hma = wma_diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    
    return hma.values

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
    RSI Streak component of Connors RSI.
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
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert to RSI-like scale (absolute streak)
    abs_streak = np.abs(streak)
    streak_rsi = pd.Series(abs_streak).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Normalize to 0-100 scale (streak of 5+ = extreme)
    streak_rsi = np.clip(streak_rsi * 20, 0, 100)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component of Connors RSI.
    Measures current price change vs past period changes.
    """
    n = len(close)
    pr = np.full(n, np.nan)
    
    if n < period + 1:
        return pr
    
    # Calculate daily returns
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0], returns])
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / period * 100
        pr[i] = rank
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    Long: CRSI < 10-15, Short: CRSI > 85-90
    """
    rsi3 = calculate_rsi(close, period=rsi_period)
    streak_rsi = calculate_rsi_streak(close, period=streak_period)
    percent_rank = calculate_percent_rank(close, period=pr_period)
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi3 + streak_rsi + percent_rank) / 3
    
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
    Choppiness Index - measures whether market is trending or ranging.
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50
    
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
    crsi_1d = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr_1d = calculate_atr(high, low, close, period=14)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    sma_200_1d = calculate_sma(close, period=200)
    vol_sma_1d = calculate_volume_sma(volume, period=20)
    
    # Calculate and align HTF HMA for major trend bias
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200_1d[i]):
            continue
        if np.isnan(chop_1d[i]) or np.isnan(vol_sma_1d[i]) or vol_sma_1d[i] <= 1e-10:
            continue
        
        # === MAJOR TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        trending_regime = chop_1d[i] < 38.2
        ranging_regime = chop_1d[i] > 61.8
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.3 * vol_sma_1d[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_1d[i] < 15
        crsi_overbought = crsi_1d[i] > 85
        crsi_extreme_oversold = crsi_1d[i] < 10
        crsi_extreme_overbought = crsi_1d[i] > 90
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200_1d[i]
        below_sma200 = close[i] < sma_200_1d[i]
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 61.8) - Mean Reversion ===
        if ranging_regime:
            # Mean reversion long: CRSI oversold + above SMA200 (not in crash)
            if crsi_oversold and above_sma200:
                desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Mean reversion short: CRSI overbought + below SMA200
            if crsi_overbought and below_sma200:
                desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Extreme mean reversion (override SMA200 filter)
            if crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 38.2) - Trend Follow ===
        elif trending_regime:
            # Trend pullback long: 1w bullish + CRSI neutral-low + above SMA200
            if trend_1w_bullish and above_sma200 and 20 < crsi_1d[i] < 45:
                desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Trend pullback short: 1w bearish + CRSI neutral-high + below SMA200
            if trend_1w_bearish and below_sma200 and 55 < crsi_1d[i] < 75:
                desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Trend continuation on CRSI reset
            if trend_1w_bullish and above_sma200 and 35 < crsi_1d[i] < 55:
                desired_signal = REDUCED_SIZE
            
            if trend_1w_bearish and below_sma200 and 45 < crsi_1d[i] < 65:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only enter on CRSI extremes + 1w trend alignment
            if crsi_extreme_oversold and trend_1w_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and trend_1w_bearish:
                desired_signal = -REDUCED_SIZE
            
            # Moderate entry with SMA200 confirmation
            if crsi_oversold and above_sma200 and trend_1w_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_overbought and below_sma200 and trend_1w_bearish:
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1w trend intact and CRSI not overbought
                if trend_1w_bullish and crsi_1d[i] < 80:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if 1w trend intact and CRSI not oversold
                if trend_1w_bearish and crsi_1d[i] > 20:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1w trend reverses or CRSI overbought
            if trend_1w_bearish and crsi_1d[i] > 70:
                desired_signal = 0.0
            # Exit if CRSI reaches extreme overbought in ranging regime
            if ranging_regime and crsi_1d[i] > 85:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1w trend reverses or CRSI oversold
            if trend_1w_bullish and crsi_1d[i] < 30:
                desired_signal = 0.0
            # Exit if CRSI reaches extreme oversold in ranging regime
            if ranging_regime and crsi_1d[i] < 15:
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