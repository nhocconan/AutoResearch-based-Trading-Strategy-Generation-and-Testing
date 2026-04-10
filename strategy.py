#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout + 1w volume confirmation + ATR stoploss
# - Long: Price breaks above Camarilla H3 level + 1w volume > 1.5x 20-period average
# - Short: Price breaks below Camarilla L3 level + same volume confirmation
# - Exit: Close-based reversal - exit long when price < Camarilla L3, exit short when price > H3
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 1d
# - Position sizing: 0.25 (discrete level)
# - Target: 30-100 total trades over 4 years (7-25/year) to stay well below HARD MAX: 150 total
# - Uses Camarilla pivots from 1d data (proven ETH edge), volume confirmation to filter false breakouts,
#   and ATR stoploss for risk management. Weekly HTF for volume ensures institutional participation.

name = "1d_1w_camarilla_pivot_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for Camarilla calculation (requires 1d OHLC)
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Pre-compute 1w data for volume
    volume_1w = df_1w['volume'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Based on previous day's OHLC
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day will have NaN due to roll
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.1*(high-low)
    # H2 = close + 0.55*(high-low)
    # H1 = close + 0.275*(high-low)
    # L1 = close - 0.275*(high-low)
    # L2 = close - 0.55*(high-low)
    # L3 = close - 1.1*(high-low)
    # L4 = close - 1.5*(high-low)
    
    high_low = prev_high - prev_low
    camarilla_h3 = prev_close + 1.1 * high_low
    camarilla_l3 = prev_close - 1.1 * high_low
    camarilla_h4 = prev_close + 1.5 * high_low
    camarilla_l4 = prev_close - 1.5 * high_low
    
    # Calculate 1w volume moving average (20-period)
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    # Calculate 1d ATR (14-period) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
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
    
    atr_14_1d = wilders_smoothing(tr, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup period (need at least 20 for volume MA)
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_14_1d[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d close
        close_price = close_1d[i]
        
        # Get current 1w volume for confirmation
        volume_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        volume_confirmation = volume_1w_current > 1.5 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H3 + volume confirmation
            if (close_price > camarilla_h3[i] and volume_confirmation):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L3 + volume confirmation
            elif (close_price < camarilla_l3[i] and volume_confirmation):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_14_1d[i]
                # Exit conditions: price < Camarilla L3 OR stoploss hit
                if close_price < camarilla_l3[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_14_1d[i]
                # Exit conditions: price > Camarilla H3 OR stoploss hit
                if close_price > camarilla_h3[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals