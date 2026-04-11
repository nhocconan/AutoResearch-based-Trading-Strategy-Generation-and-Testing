#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Confirmation
# - Williams Alligator: Jaw(13), Teeth(8), Lips(5) SMAs on median price
# - Bullish: Lips > Teeth > Jaw (alligator eating up)
# - Bearish: Lips < Teeth < Jaw (alligator eating down)
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Long: Bullish Alligator + Bull Power > 0 + Volume > 1.5x 20-period avg
# - Short: Bearish Alligator + Bear Power < 0 + Volume > 1.5x 20-period avg
# - Exit: Opposite Alligator alignment or power crosses zero
# - Uses 1w trend filter: price > EMA(50) for long bias, price < EMA(50) for short bias
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Works in both bull and bear by capturing momentum with Alligator alignment
# - Volume confirmation ensures breakouts have conviction
# - Elder Ray adds bull/bear power confirmation

name = "12h_1w_alligator_elder_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Median price for Alligator: (high + low) / 2
    median_price = (high + low) / 2
    
    # Williams Alligator SMAs
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Elder Ray: EMA(13) of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 20-period average
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Alligator alignment
        bullish_alligator = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alligator = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray power
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1w EMA trend bias
        ema_bias_long = close_price > ema_50_1w_aligned[i]
        ema_bias_short = close_price < ema_50_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bullish Alligator + Bull Power positive + Volume confirmation + Long bias
        if bullish_alligator and bull_power_positive and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short: Bearish Alligator + Bear Power negative + Volume confirmation + Short bias
        if bearish_alligator and bear_power_negative and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Bearish Alligator forms or Bull Power turns negative
            exit_long = bearish_alligator or not bull_power_positive
        elif position == -1:
            # Exit short if Bullish Alligator forms or Bear Power turns positive
            exit_short = bullish_alligator or not bear_power_negative
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals