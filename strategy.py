#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend with 1d RSI mean reversion and volume spike
# - Long when 4h KAMA is rising AND 1d RSI < 30 (oversold) AND 1d volume > 1.8x 20-period volume SMA
# - Short when 4h KAMA is falling AND 1d RSI > 70 (overbought) AND 1d volume > 1.8x 20-period volume SMA
# - Exit: ATR trailing stop (2.0*ATR) from highest/lowest since entry
# - Uses 4h for trend direction (KAMA adapts to market noise), 1d for RSI extremes and volume confirmation
# - KAMA reduces whipsaw in choppy markets; RSI mean reversion captures reversals in extended moves
# - Volume spike confirms institutional participation; tight entries target ~30-60 trades/year
# - Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) with volume filter

name = "4h_1d_kama_rsi_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for RSI and volume (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing for RSI
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            result[period-1] = np.nanmean(values[:period])
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    avg_gain = wilders_smoothing(gain, 14)
    avg_loss = wilders_smoothing(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute KAMA on 4h (primary timeframe)
    # KAMA parameters: ER period=10, Fast EMA=2, Slow EMA=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[:1])  # placeholder, will compute properly below
    
    # Proper efficiency ratio calculation
    dir = np.abs(np.diff(close, lookback=10)) if hasattr(np.diff, '__kwdefaults__') else np.abs(close[10:] - close[:-10])
    # Manual calculation for efficiency ratio
    change_over_period = np.zeros(n)
    volatility_sum = np.zeros(n)
    lookback_er = 10
    
    for i in range(lookback_er, n):
        change_over_period[i] = np.abs(close[i] - close[i - lookback_er])
        volatility_sum[i] = np.sum(np.abs(np.diff(close[i - lookback_er:i+1])))
    
    # Avoid division by zero
    er = np.where(volatility_sum > 0, change_over_period / volatility_sum, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction: rising if current > previous, falling if current < previous
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    kama_rising[0] = False
    kama_falling[0] = False
    
    # ATR for dynamic stoploss (using 4h data)
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Track highest/lowest since entry for trailing stop
    highest_high_since_entry = np.full(n, np.nan)
    lowest_low_since_entry = np.full(n, np.nan)
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.8x 20-period volume SMA
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_confirm = vol_1d_aligned[i] > 1.8 * volume_sma_20_1d_aligned[i]
        
        # Only trade when volume confirmation is present
        if vol_confirm:
            # Long: KAMA rising AND RSI < 30 (oversold)
            if kama_rising[i] and rsi_1d_aligned[i] < 30:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                    highest_high_since_entry[i] = high[i]
                else:
                    signals[i] = 0.25
                    highest_high_since_entry[i] = max(highest_high_since_entry[i-1] if i > 0 else high[i], high[i])
            # Short: KAMA falling AND RSI > 70 (overbought)
            elif kama_falling[i] and rsi_1d_aligned[i] > 70:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                    lowest_low_since_entry[i] = low[i]
                else:
                    signals[i] = -0.25
                    lowest_low_since_entry[i] = min(lowest_low_since_entry[i-1] if i > 0 else low[i], low[i])
            else:
                # Maintain position and update tracking levels
                if position == 1:
                    signals[i] = 0.25
                    highest_high_since_entry[i] = max(highest_high_since_entry[i-1] if i > 0 else high[i], high[i])
                elif position == -1:
                    signals[i] = -0.25
                    lowest_low_since_entry[i] = min(lowest_low_since_entry[i-1] if i > 0 else low[i], low[i])
                else:
                    signals[i] = 0.0
            
            # Check for ATR trailing stop exit
            if position == 1 and not np.isnan(highest_high_since_entry[i]):
                if close[i] < (highest_high_since_entry[i] - 2.0 * atr_4h[i]):
                    position = 0
                    signals[i] = 0.0
            elif position == -1 and not np.isnan(lowest_low_since_entry[i]):
                if close[i] > (lowest_low_since_entry[i] + 2.0 * atr_4h[i]):
                    position = 0
                    signals[i] = 0.0
        else:
            # No trade: exit any position if conditions not met
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals