#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses 12h timeframe for signal generation with Camarilla pivot levels from 1d data
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# 1d EMA34 > 1d EMA89 filter for bull/bear regime alignment
# Discrete position sizing (0.25) balances return and risk while minimizing fee drag
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Camarilla levels provide high-probability reversal/breakout zones
# Works in both bull and bear markets by aligning with 1d trend direction

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and EMA filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 and EMA89 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = close_1d.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d EMAs to 12h timeframe (wait for completed 1d candle)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema89_1d)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla formula: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3_1d = close_1d_prev + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3_1d = close_1d_prev - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume confirmation (2.0x 20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema89_1d_aligned[i]) or 
            np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA alignment
        bullish_trend = ema34_1d_aligned[i] > ema89_1d_aligned[i]
        bearish_trend = ema34_1d_aligned[i] < ema89_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Camarilla R3 + volume confirm + bullish 1d trend
            if close[i] > camarilla_r3_1d_aligned[i] and volume_confirm[i] and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: Close < Camarilla S3 + volume confirm + bearish 1d trend
            elif close[i] < camarilla_s3_1d_aligned[i] and volume_confirm[i] and bearish_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Camarilla S3 (reversal to downside) or trend turns bearish
            if close[i] < camarilla_s3_1d_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close > Camarilla R3 (reversal to upside) or trend turns bullish
            if close[i] > camarilla_r3_1d_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses 12h timeframe for signal generation with Camarilla pivot levels from 1d data
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# 1d EMA34 > 1d EMA89 filter for bull/bear regime alignment
# Discrete position sizing (0.25) balances return and risk while minimizing fee drag
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Camarilla levels provide high-probability reversal/breakout zones
# Works in both bull and bear markets by aligning with 1d trend direction

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and EMA filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 and EMA89 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = close_1d.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d EMAs to 12h timeframe (wait for completed 1d candle)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema89_1d)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla formula: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3_1d = close_1d_prev + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3_1d = close_1d_prev - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume confirmation (2.0x 20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema89_1d_aligned[i]) or 
            np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA alignment
        bullish_trend = ema34_1d_aligned[i] > ema89_1d_aligned[i]
        bearish_trend = ema34_1d_aligned[i] < ema89_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Camarilla R3 + volume confirm + bullish 1d trend
            if close[i] > camarilla_r3_1d_aligned[i] and volume_confirm[i] and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: Close < Camarilla S3 + volume confirm + bearish 1d trend
            elif close[i] < camarilla_s3_1d_aligned[i] and volume_confirm[i] and bearish_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Camarilla S3 (reversal to downside) or trend turns bearish
            if close[i] < camarilla_s3_1d_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close > Camarilla R3 (reversal to upside) or trend turns bullish
            if close[i] > camarilla_r3_1d_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals