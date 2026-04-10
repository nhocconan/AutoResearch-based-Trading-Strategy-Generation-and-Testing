#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level in 1w uptrend (close > EMA34) with volume > 1.8x 20-bar avg
# - Short when price breaks below Camarilla L3 level in 1w downtrend (close < EMA34) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~15 trades/year (60 total over 4 years) to avoid fee drag
# - 1w trend filter reduces false breakouts in ranging markets
# - ATR-based stoploss to limit drawdown

name = "1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1w volume confirmation: > 1.8x 20-period average
    avg_volume_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (1.8 * avg_volume_20_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # Pre-compute ATR for stoploss (using 1d data)
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_ranges = np.nanmax(ranges.values, axis=1)
    atr_14 = pd.Series(true_ranges).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop_multiplier = 2.5
    
    for i in range(50, n):  # Need enough data for Camarilla calculation
        # Skip if any required data is invalid
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_spike_1w_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # ATR-based stoploss
            if prices['close'].iloc[i] < entry_price - atr_stop_multiplier * atr_14[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # ATR-based stoploss
            if prices['close'].iloc[i] > entry_price + atr_stop_multiplier * atr_14[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate Camarilla pivot levels from previous day (using previous completed bar)
            if i >= 1:
                prev_high = prices['high'].iloc[i-1]
                prev_low = prices['low'].iloc[i-1]
                prev_close = prices['close'].iloc[i-1]
                
                # Camarilla pivot levels
                pivot = (prev_high + prev_low + prev_close) / 3
                range_val = prev_high - prev_low
                
                # Resistance levels
                h3 = pivot + (range_val * 1.1 / 2)
                h4 = pivot + (range_val * 1.1)
                
                # Support levels
                l3 = pivot - (range_val * 1.1 / 2)
                l4 = pivot - (range_val * 1.1)
                
                # Long signal: price breaks above Camarilla H3 in 1w uptrend with volume spike
                if (prices['close'].iloc[i] > h3 and 
                    prices['close'].iloc[i] > ema_34_1w_aligned[i] and 
                    vol_spike_1w_aligned[i]):
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Camarilla L3 in 1w downtrend with volume spike
                elif (prices['close'].iloc[i] < l3 and 
                      prices['close'].iloc[i] < ema_34_1w_aligned[i] and 
                      vol_spike_1w_aligned[i]):
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals