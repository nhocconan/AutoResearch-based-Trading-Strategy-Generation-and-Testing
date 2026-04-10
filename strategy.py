#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d trend filter
# - Uses 1d Camarilla levels (R3/S3, R4/S4) for breakout/fade logic
# - Long when price breaks above R4 with 12h volume > 2.0x 20-bar avg AND 1d close > 1d EMA50
# - Short when price breaks below S4 with 12h volume > 2.0x 20-bar avg AND 1d close < 1d EMA50
# - Exit when price retouches the 1d VWAP (mean reversion to fair value)
# - Camarilla pivots work well in both ranging and trending markets
# - Volume confirmation reduces false breakouts
# - 1d EMA50 filter ensures trades align with higher timeframe trend
# - Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years)

name = "6h_12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 50 or len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # where C = (H+L+Close)/3 (typical price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    hl_range = df_1d['high'] - df_1d['low']
    
    # Previous day's typical price and range (shifted by 1 to avoid look-ahead)
    prev_typical = typical_price.shift(1)
    prev_hl = hl_range.shift(1)
    
    camarilla_r4 = prev_typical + (prev_hl * 1.1 / 2)
    camarilla_r3 = prev_typical + (prev_hl * 1.1 / 4)
    camarilla_s3 = prev_typical - (prev_hl * 1.1 / 4)
    camarilla_s4 = prev_typical - (prev_hl * 1.1 / 2)
    
    # Pre-compute 1d VWAP for exit (typical price * volume / cumulative volume)
    # Using typical price approximation for VWAP
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_num = (typical_price_1d * df_1d['volume']).cumsum()
    vwap_den = df_1d['volume'].cumsum()
    vwap_1d = vwap_num / vwap_den
    # Handle division by zero on first bar
    vwap_1d = vwap_1d.fillna(typical_price_1d)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = df_1d['close'].ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Pre-compute 12h volume confirmation: > 2.0x 20-period average
    volume_20_avg_12h = df_12h['volume'].rolling(window=20, min_periods=20).mean()
    vol_spike_12h = df_12h['volume'] > (2.0 * volume_20_avg_12h)
    
    # Align all HTF data to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4.values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4.values)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d.values)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d.values)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h.values, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vwap_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_price = prices['close'].iloc[i]
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > R4 with volume spike and 1d uptrend
            if (close_price > camarilla_r4_aligned[i] and
                vol_spike_12h_aligned[i] and
                close_price > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < S4 with volume spike and 1d downtrend
            elif (close_price < camarilla_s4_aligned[i] and
                  vol_spike_12h_aligned[i] and
                  close_price < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at VWAP (mean reversion)
            # Exit when price retouches 1d VWAP
            exit_signal = False
            if position == 1:  # Long position
                if close_price <= vwap_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if close_price >= vwap_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals