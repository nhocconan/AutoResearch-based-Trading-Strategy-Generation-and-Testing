#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R1 AND 1d close > EMA34 (bullish trend) AND volume > 2.0x 20-bar average.
# Short when price breaks below S1 AND 1d close < EMA34 (bearish trend) AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 75-200 total trades over 4 years (19-50/year).
# Camarilla pivot levels provide institutional price structure, EMA34 filters higher timeframe trend,
# volume spike confirms participation. Designed to work in both bull and bear markets via trend filter.

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels from previous day
    # Need previous day's high, low, close
    prices_df = prices.copy()
    prices_df['date'] = prices_df['open_time'].dt.date
    unique_dates = prices_df['date'].unique()
    
    # Map each 4h bar to its corresponding Camarilla levels from previous day
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(1, len(unique_dates)):
        prev_date = unique_dates[i-1]
        curr_date = unique_dates[i]
        
        # Get previous day's OHLC
        prev_day = prices_df[prices_df['date'] == prev_date]
        if len(prev_day) == 0:
            continue
            
        ph = prev_day['high'].max()
        pl = prev_day['low'].min()
        pc = prev_day['close'].iloc[-1]
        
        # Calculate Camarilla levels
        camarilla_range = ph - pl
        r1 = pc + (camarilla_range * 1.1 / 12)
        s1 = pc - (camarilla_range * 1.1 / 12)
        
        # Apply to current day's 4h bars
        curr_day_bars = prices_df[prices_df['date'] == curr_date].index
        for idx in curr_day_bars:
            if idx < n:
                camarilla_r1[idx] = r1
                camarilla_s1[idx] = s1
    
    # 1d EMA34 trend filter
    prev_close_1d = df_1d['close'].values
    ema_34 = pd.Series(prev_close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: current 4h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
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
        breakout_above_r1 = curr_high > camarilla_r1[i]  # Using high for breakout confirmation
        breakout_below_s1 = curr_low < camarilla_s1[i]   # Using low for breakout confirmation
        
        # Trend filter: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = curr_close > ema_34_aligned[i]
        bearish_trend = curr_close < ema_34_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R1 AND bullish trend AND volume confirmation
            if (breakout_above_r1 and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 AND bearish trend AND volume confirmation
            elif (breakout_below_s1 and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S1 (mean reversion) OR trend turns bearish
            if (curr_low < camarilla_s1[i] or 
                bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above R1 (mean reversion) OR trend turns bullish
            if (curr_high > camarilla_r1[i] or 
                bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals