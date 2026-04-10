#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w volume confirmation and ATR-based stoploss
# - Long: Price breaks above Camarilla H4 level + 1w volume > 1.5x 20-period average volume
# - Short: Price breaks below Camarilla L4 level + same volume confirmation
# - Exit: Close-based reversal - exit long when price < L3, exit short when price > H3
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 12h
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) to stay within HARD MAX: 200 total
# - Uses Camarilla pivot levels from 12h data, volume confirmation from weekly HTF to filter false breakouts,
#   and ATR stoploss for risk management. Weekly HTF volume ensures institutional participation.
#   Works in both bull (breakouts) and bear (mean reversion at extremes) regimes.

name = "12h_1w_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need enough data for calculation
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Pre-compute 1w data for volume
    volume_1w = df_1w['volume'].values
    
    # Calculate Camarilla pivot levels for 12h (based on previous bar)
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = high_12h[0]  # First bar uses current values
    prev_low[0] = low_12h[0]
    prev_close[0] = close_12h[0]
    
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Calculate 1w volume moving average (20-period)
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    # Calculate 12h ATR (14-period) for stoploss
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First TR is 0 (no previous close)
    
    # Wilder's ATR smoothing
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])  # First value is simple average
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_14_12h = wilders_smoothing(tr, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup period (need at least 20 for ATR and volume MA)
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_14_12h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h close
        close_price = close_12h[i]
        
        # Get current 1w volume for confirmation
        volume_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        volume_confirmation = volume_1w_current > 1.5 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H4 + volume confirmation
            if (close_price > camarilla_h4[i] and volume_confirmation):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L4 + volume confirmation
            elif (close_price < camarilla_l4[i] and volume_confirmation):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_14_12h[i]
                # Exit conditions: price < L3 OR stoploss hit
                if close_price < camarilla_l3[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_14_12h[i]
                # Exit conditions: price > H3 OR stoploss hit
                if close_price > camarilla_h3[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals