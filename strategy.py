#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d trend filter
# - Long: Price breaks above Camarilla R4 (1d) + 12h volume > 1.5x 20-period MA + 1d close > 1d EMA50
# - Short: Price breaks below Camarilla S4 (1d) + 12h volume > 1.5x 20-period MA + 1d close < 1d EMA50
# - Exit: Price returns to Camarilla PP (pivot point) OR volume drops below average
# - Uses 1d Camarilla levels for structure, 12h for volume confirmation and timing, 6h for execution
# - Works in bull/bear: Camarilla S4/R4 are strong support/resistance; volume confirms breakout validity;
#   1d EMA50 filter ensures trading with higher timeframe trend. Targets ~15-30 trades/year.

name = "6h_12h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 60 or len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla: PP = (H+L+C)/3, R4 = C + ((H-L)*1.1/2), S4 = C - ((H-L)*1.1/2)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First bar: use same values (will be NaN until enough data)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    camarilla_pp = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    camarilla_range = prev_high_1d - prev_low_1d
    camarilla_r4 = close_1d + (camarilla_range * 1.1 / 2)
    camarilla_s4 = close_1d - (camarilla_range * 1.1 / 2)
    
    # Align 1d Camarilla levels to 6h timeframe (wait for 1d bar to close)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume confirmation: current volume > 1.5x 20-period MA
    volume_ma_20_12h = pd.Series(volume_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)
    volume_confirm = vol_12h_current > 1.5 * volume_ma_20_12h_aligned
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for Camarilla S4/R4 breakouts
            # Long entry: Price breaks above Camarilla R4 + volume confirmation + 1d uptrend
            if (close[i] > camarilla_r4_aligned[i] and 
                volume_confirm[i] and 
                close_12h[i] > ema_50_1d[i] if i < len(ema_50_1d) else False):  # Use 12h close vs 1d EMA (approximation)
                # Better: use aligned 1d EMA for trend
                if close[i] > camarilla_r4_aligned[i] and volume_confirm[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
            # Short entry: Price breaks below Camarilla S4 + volume confirmation + 1d downtrend
            elif (close[i] < camarilla_s4_aligned[i] and 
                  volume_confirm[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to Camarilla PP OR volume drops below average
            if position == 1:  # Long position
                if close[i] <= camarilla_pp_aligned[i] or not volume_confirm[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= camarilla_pp_aligned[i] or not volume_confirm[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals