#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
# Long when price is above Ichimoku cloud (Senkou Span A/B) with Tenkan > Kijun and 1d EMA34 uptrend.
# Short when price is below cloud with Tenkan < Kijun and 1d EMA34 downtrend.
# Uses volume confirmation: current volume > 1.5x 6h EMA20 volume average.
# Discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Ichimoku components calculated from 6h data. 1d EMA34 for trend filter loaded once.
# Works in bull (cloud acts as support/resistance) and bear (cloud filters false breakouts).

name = "6h_Ichimoku_Cloud_1dEMA34_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max()
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 plotted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Load 1d data ONCE before loop for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe (waits for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume EMA20 for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Ichimoku (max 52 periods) and ATR
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(tenkan.iloc[i]) or 
            np.isnan(kijun.iloc[i]) or 
            np.isnan(senkou_a.iloc[i]) or 
            np.isnan(senkou_b.iloc[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Ichimoku values at current bar
        tenkan_val = tenkan.iloc[i]
        kijun_val = kijun.iloc[i]
        senkou_a_val = senkou_a.iloc[i]
        senkou_b_val = senkou_b.iloc[i]
        
        # Cloud top and bottom (Senkou Span A/B)
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Volume confirmation: current volume > 1.5x 6h EMA20 volume average
        volume_confirm = curr_volume > (vol_ema_20[i] * 1.5)
        
        # Trend filter: 1d EMA34 direction (using slope over 3 bars)
        if i >= 3:
            ema_slope = ema_34_1d_aligned[i] - ema_34_1d_aligned[i-3]
            trend_up = ema_slope > 0
            trend_down = ema_slope < 0
        else:
            trend_up = False
            trend_down = False
        
        # Ichimoku signals
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        tenkan_above_kijun = tenkan_val > kijun_val
        tenkan_below_kijun = tenkan_val < kijun_val
        
        if position == 0:  # Flat - look for new entries
            # Long: price above cloud, Tenkan > Kijun, volume confirm, 1d EMA34 up
            if (price_above_cloud and 
                tenkan_above_kijun and 
                volume_confirm and 
                trend_up):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price below cloud, Tenkan < Kijun, volume confirm, 1d EMA34 down
            elif (price_below_cloud and 
                  tenkan_below_kijun and 
                  volume_confirm and 
                  trend_down):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price falls below cloud OR Tenkan < Kijun
            elif (curr_close < cloud_top) or (tenkan_val < kijun_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price rises above cloud OR Tenkan > Kijun
            elif (curr_close > cloud_bottom) or (tenkan_val > kijun_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals