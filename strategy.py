#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 AND 1w close > EMA34 (bullish trend) AND volume > 2.0x 20-bar average.
# Short when price breaks below S3 AND 1w close < EMA34 (bearish trend) AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.30 to manage drawdown. Target: 30-100 total trades over 4 years (7-25/year).
# Camarilla levels provide institutional support/resistance, EMA34 filters trend alignment, volume spike confirms breakout strength.
# Primary timeframe: 1d, HTF: 1w for EMA trend filter.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from previous day
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2 where C=(H+L+Close)/3
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first bar has no previous
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    rang = prev_high - prev_low
    r3 = pivot + (rang * 1.1 / 2.0)
    s3 = pivot - (rang * 1.1 / 2.0)
    
    # 1w EMA34 trend filter
    prev_close_1w = df_1w['close'].values
    ema_34 = pd.Series(prev_close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume confirmation: current 1d volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(r3[i]) or np.isnan(s3[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Breakout conditions
        breakout_long = curr_high > r3[i]   # price breaks above R3
        breakout_short = curr_low < s3[i]   # price breaks below S3
        
        # Trend filter: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = curr_close > ema_34_aligned[i]
        bearish_trend = curr_close < ema_34_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND bullish trend AND volume confirmation
            if (breakout_long and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.30
                position = 1
            # Short: breakout below S3 AND bearish trend AND volume confirmation
            elif (breakout_short and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below pivot (mean reversion) OR trend turns bearish
            if (curr_close < pivot[i] or 
                bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price crosses above pivot (mean reversion) OR trend turns bullish
            if (curr_close > pivot[i] or 
                bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals