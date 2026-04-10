#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs of median price
# - Long when Lips > Teeth > Jaw (bullish alignment) AND 1w close > 1w EMA(20) AND 1d volume > 1.5x 20-bar avg
# - Short when Lips < Teeth < Jaw (bearish alignment) AND 1w close < 1w EMA(20) AND 1d volume > 1.5x 20-bar avg
# - Exit when Alligator lines re-cross (Lips crosses Teeth) indicating trend weakening
# - Uses discrete position sizing (0.25) to control risk and minimize fee churn
# - Williams Alligator identifies trending vs ranging markets via convergence/divergence
# - 1w EMA filter ensures alignment with weekly trend to avoid counter-trend trades
# - Volume confirmation avoids low-liquidity false signals
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)

name = "1d_1w_williams_alligator_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    weekly_bullish = close_1w > ema_20_1w
    weekly_bearish = close_1w < ema_20_1w
    
    # Pre-compute 1d Williams Alligator
    # Median price = (high + low + close) / 3
    median_price = (prices['high'] + prices['low'] + prices['close']) / 3
    median_price_vals = median_price.values
    
    # Jaw: 13-period SMA, smoothed by 8 periods
    jaw_raw = pd.Series(median_price_vals).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMA, smoothed by 5 periods
    teeth_raw = pd.Series(median_price_vals).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMA, smoothed by 3 periods
    lips_raw = pd.Series(median_price_vals).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # Alligator alignment signals
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (lips < teeth) & (teeth < jaw)
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = prices['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Align HTF indicators to 1d timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    # Align Alligator components to 1d (though calculated on 1d, we align for consistency)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)  # Using 1w as dummy HTF for alignment structure
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Recalculate alignment properly using 1d data for Alligator
    # Since Alligator is calculated on 1d, we need to align using 1d as both LT and HTF
    # For simplicity, we'll use the directly calculated values since they're already on 1d
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    bullish_alignment_aligned = bullish_alignment
    bearish_alignment_aligned = bearish_alignment
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(vol_spike_1d[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new Alligator alignment entries
            # Long when bullish alignment AND weekly bullish trend AND volume spike
            if (bullish_alignment_aligned[i] and 
                weekly_bullish_aligned[i] and 
                vol_spike_1d[i]):
                position = 1
                signals[i] = 0.25
            # Short when bearish alignment AND weekly bearish trend AND volume spike
            elif (bearish_alignment_aligned[i] and 
                  weekly_bearish_aligned[i] and 
                  vol_spike_1d[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when Alligator re-crosses
            # Exit when Lips crosses Teeth (trend weakening)
            exit_long = position == 1 and lips_aligned[i] < teeth_aligned[i]
            exit_short = position == -1 and lips_aligned[i] > teeth_aligned[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals