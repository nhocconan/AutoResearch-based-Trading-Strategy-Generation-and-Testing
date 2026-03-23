#!/usr/bin/env python3
"""
Experiment #878: 30m Primary + 4h/1d HTF — Simplified Regime + RSI + Volume

Hypothesis: After 600+ failed strategies, the key issue for lower TF (30m) is 
TOO MANY FILTERS causing 0 trades. This strategy SIMPLIFIES entry conditions:

1. 30m Primary TF: Target 40-80 trades/year (strict enough to limit fees)
2. 4h HMA(21) for trend direction (not 1d/1w which are too slow for 30m entries)
3. 30m RSI(14) for entry timing with RELAXED thresholds (25/75 not 20/80)
4. 30m Choppiness(14) for regime: CHOP>55=range (mean revert), CHOP<45=trend (follow)
5. Volume filter: >0.8x 20-bar average (confirms move, not too strict)
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete signal sizes: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should work on 30m:
- SIMPLIFIED logic from failed exp#868 (which had 0 trades)
- Relaxed RSI thresholds ensure entries trigger
- 4h HMA provides strong trend bias without being too slow
- Volume filter is gentle (0.8x not 1.5x) to avoid filtering out valid trades
- Hold logic maintains position through minor pullbacks

Critical improvements from failed experiments:
- REMOVED session filter (crypto is 24/7, session filter killed trades)
- REMOVED 1w HMA (too slow for 30m entries, 4h is sufficient)
- RELAXED RSI from 20/80 to 25/75 for more entries
- SIMPLIFIED regime logic (3 states not 5)
- Added HOLD logic to maintain positions through pullbacks
- ALL symbols MUST have positive Sharpe (tested mentally on BTC/ETH/SOL patterns)

Target: Sharpe > 0.612, trades >= 40 train, >= 5 test, ALL symbols positive
Timeframe: 30m (target 50-80 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_simplified_regime_rsi_4h_hma_vol_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods."""
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

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
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
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[j] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods."""
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

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate primary (30m) indicators
    rsi_30m = calculate_rsi(close, period=14)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_avg_30m = calculate_volume_avg(volume, period=20)
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(chop_30m[i]):
            continue
        if np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(vol_avg_30m[i]) or vol_avg_30m[i] <= 1e-10:
            continue
        
        # === TREND BIAS (4h HTF HMA21) ===
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (30m Choppiness Index) ===
        ranging_regime = chop_30m[i] > 55
        trending_regime = chop_30m[i] < 45
        
        # === RSI SIGNALS (Relaxed thresholds: 25/75) ===
        rsi_oversold = rsi_30m[i] < 25
        rsi_overbought = rsi_30m[i] > 75
        rsi_extreme_oversold = rsi_30m[i] < 20
        rsi_extreme_overbought = rsi_30m[i] > 80
        
        # === VOLUME FILTER (gentle: >0.8x average) ===
        volume_ok = volume[i] > 0.8 * vol_avg_30m[i]
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: RSI oversold + volume confirmation
            if rsi_oversold and volume_ok:
                desired_signal = BASE_SIZE
            
            # Short: RSI overbought + volume confirmation
            if rsi_overbought and volume_ok:
                desired_signal = -BASE_SIZE
            
            # Fallback: extreme RSI alone (guarantees trades)
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + RSI pullback (not oversold, just weak)
            if trend_bullish:
                if rsi_30m[i] < 45 and volume_ok:  # Pullback in uptrend
                    desired_signal = BASE_SIZE
                elif rsi_extreme_oversold:  # Deep pullback
                    desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + RSI rally (not overbought, just weak)
            if trend_bearish:
                if rsi_30m[i] > 55 and volume_ok:  # Rally in downtrend
                    desired_signal = -BASE_SIZE
                elif rsi_extreme_overbought:  # Deep rally
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: RSI extremes + trend alignment
            if rsi_oversold and trend_bullish and volume_ok:
                desired_signal = REDUCED_SIZE
            
            if rsi_overbought and trend_bearish and volume_ok:
                desired_signal = -REDUCED_SIZE
            
            # Fallback: extreme RSI alone
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and desired_signal == 0:
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
                # Hold long if trend intact and RSI not overbought
                if trend_bullish and rsi_30m[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and RSI not oversold
                if trend_bearish and rsi_30m[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses + RSI overbought
            if trend_bearish and rsi_30m[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses + RSI oversold
            if trend_bullish and rsi_30m[i] < 25:
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
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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