#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA trend filter and volume spike confirmation.
# Long when Williams %R(14) crosses above -80 (oversold) AND price > 1d EMA34 AND 6h volume > 1.5x 20-period average.
# Short when Williams %R(14) crosses below -20 (overbought) AND price < 1d EMA34 AND 6h volume > 1.5x 20-period average.
# Exit when Williams %R returns to -50 (mean reversion midpoint) or opposite extreme is reached.
# Uses discrete position size 0.25. Williams %R identifies exhaustion points, EMA34 filters trend direction,
# volume spike confirms momentum behind the move. Target: 60-120 total trades over 4 years (15-30/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Williams %R(14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # === 6h Indicators: Volume MA(20) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data once before loop for EMA34 filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA(34) for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        wr = williams_r[i]
        vol_ma_val = vol_ma_20[i]
        ema_val = ema_34_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.5x 20-period average
        vol_filter = vol > 1.5 * vol_ma_val if vol_ma_val > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R returns to -50 (mean reversion) or reaches -10 (overbought)
            if wr >= -50 or wr >= -10:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R returns to -50 (mean reversion) or reaches -90 (oversold)
            if wr <= -50 or wr <= -90:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R crosses above -80 (from below) with volume and trend confirmation
            if wr > -80 and wr <= -80 + 0.1 and price > ema_val and vol_filter:  # crossed above -80
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R crosses below -20 (from above) with volume and trend confirmation
            elif wr < -20 and wr >= -20 - 0.1 and price < ema_val and vol_filter:  # crossed below -20
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_WilliamsR_MeanReversion_1dEMA34_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0