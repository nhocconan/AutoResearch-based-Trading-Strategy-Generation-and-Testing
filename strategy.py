#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d trend filter
# - Long when price breaks above Camarilla R4 AND 12h volume > 1.3x 20-period volume SMA AND 1d close > 1d EMA50
# - Short when price breaks below Camarilla S4 AND 12h volume > 1.3x 20-period volume SMA AND 1d close < 1d EMA50
# - Exit: price returns to Camarilla pivot point (PP) or volume drops below average
# - Uses 6h for Camarilla calculation and breakout, 12h for volume confirmation, 1d for trend filter
# - Camarilla levels provide intraday support/resistance with statistical edge
# - Volume confirmation ensures breakouts have institutional participation
# - 1d EMA50 filter avoids counter-trend trades in strong trends
# - Target: 12-30 trades/year to minimize fee drag while capturing high-probability breakouts

name = "6h_12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
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
    
    # Load 12h data ONCE before loop for volume confirmation (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Load 1d data ONCE before loop for trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute Camarilla pivot levels for 6h data (using prior bar's OHLC)
    # Camarilla calculations based on prior period's range
    prior_close = np.roll(close, 1)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close[0] = close[0]  # First bar uses current values
    prior_high[0] = high[0]
    prior_low[0] = low[0]
    
    # Calculate pivot point and ranges
    pivot_point = (prior_high + prior_low + prior_close) / 3.0
    range_hl = prior_high - prior_low
    
    # Camarilla levels
    camarilla_pp = pivot_point
    camarilla_r1 = pivot_point + (range_hl * 1.1 / 12)
    camarilla_r2 = pivot_point + (range_hl * 1.1 / 6)
    camarilla_r3 = pivot_point + (range_hl * 1.1 / 4)
    camarilla_r4 = pivot_point + (range_hl * 1.1 / 2)
    camarilla_s1 = pivot_point - (range_hl * 1.1 / 12)
    camarilla_s2 = pivot_point - (range_hl * 1.1 / 6)
    camarilla_s3 = pivot_point - (range_hl * 1.1 / 4)
    camarilla_s4 = pivot_point - (range_hl * 1.1 / 2)
    
    # Pre-compute volume SMA for 12h data (20-period)
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute EMA50 for 1d close (trend filter)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d close aligned for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    for i in range(20, n):  # Start after 20-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or 
            np.isnan(camarilla_pp[i]) or np.isnan(volume_sma_20_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.3x 20-period volume SMA
        # Get current 12h volume (need to align it)
        vol_12h_current = df_12h['volume'].values
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h_current)
        vol_confirm = vol_12h_aligned[i] > 1.3 * volume_sma_20_12h_aligned[i]
        
        # Trend filter: 1d close vs EMA50
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Camarilla breakout signals
        breakout_long = close[i] > camarilla_r4[i]  # Break above R4
        breakout_short = close[i] < camarilla_s4[i]  # Break below S4
        
        # Exit condition: price returns to pivot point
        exit_long = close[i] < camarilla_pp[i]
        exit_short = close[i] > camarilla_pp[i]
        
        # Trading logic
        if vol_confirm:
            # Long: Camarilla R4 breakout in uptrend
            if breakout_long and uptrend:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: Camarilla S4 breakdown in downtrend
            elif breakout_short and downtrend:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits
                if position == 1 and exit_long:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals