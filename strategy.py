#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Uses Ichimoku cloud (senkou span A/B) from 6h timeframe for dynamic support/resistance
# Entry: price breaks above/below cloud with TK cross confirmation and 1d EMA50 trend alignment
# Exit: price re-enters cloud or TK cross reverses
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in bull/bear via 1d EMA50 trend filter - only trades in direction of daily momentum
# Ichimoku provides adaptive trend/filter that performs well in ranging and trending markets

name = "6h_Ichimoku_Cloud_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # need enough data for Ichimoku calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind (not used for signals)
    
    # Align Ichimoku components (they are already calculated on 6h timeframe)
    # Senkou spans need to be shifted forward by 26 periods for cloud plotting
    # But for signal generation, we use current cloud values (already shifted in calculation)
    # Actually, senkou_a/b as calculated above are the values to be plotted 26 periods ahead
    # So current cloud is senkou_a/b from 26 periods ago
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # Set first 26 values to NaN since we don't have cloud data yet
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(52, 26) + 26  # need senkou b period + cloud shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(senkou_a_lagged[i]) or 
            np.isnan(senkou_b_lagged[i]) or np.isnan(tenkan[i]) or 
            np.isnan(kijun[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        curr_senkou_a = senkou_a_lagged[i]
        curr_senkou_b = senkou_b_lagged[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Cloud boundaries (top and bottom of cloud)
        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        
        # TK cross (Tenkan crossing Kijun)
        tk_cross_bullish = curr_tenkan > curr_kijun
        tk_cross_bearish = curr_tenkan < curr_kijun
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price re-enters cloud or TK cross turns bearish
            if (curr_close < entry_price - 2.0 * atr_at_entry or 
                curr_close < cloud_top or 
                not tk_cross_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price re-enters cloud or TK cross turns bullish
            if (curr_close > entry_price + 2.0 * atr_at_entry or 
                curr_close > cloud_bottom or 
                not tk_cross_bearish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new breakout entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long breakout when price breaks above cloud with bullish TK cross and 1d EMA50 uptrend
            if (curr_close > cloud_top and 
                tk_cross_bullish and 
                curr_close > curr_ema50_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short breakout when price breaks below cloud with bearish TK cross and 1d EMA50 downtrend
            elif (curr_close < cloud_bottom and 
                  tk_cross_bearish and 
                  curr_close < curr_ema50_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals