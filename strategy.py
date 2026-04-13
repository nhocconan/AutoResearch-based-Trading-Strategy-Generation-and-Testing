#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Ichimoku cloud with weekly trend filter and volume confirmation
    # Ichimoku captures trend/momentum/cloud support/resistance
    # Weekly trend (EMA 200) filters false signals in choppy markets
    # Volume confirmation ensures breakouts have participation
    # Works in bull/bear: cloud acts as dynamic S/R, weekly trend avoids counter-trend trades
    # Target: 12-30 trades/year to minimize fee drag on 6h timeframe
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 6h data for Ichimoku calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 60:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get weekly data for trend filter (EMA 200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    daily_volume = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((max_high_52 + min_low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used for entry as it requires future data
    
    # Align Ichimoku components to 6h primary timeframe (no shift needed as calculated on 6h)
    tenkan_aligned = tenkan_sen
    kijun_aligned = kijun_sen
    senkou_a_aligned = senkou_span_a
    senkou_b_aligned = senkou_span_b
    
    # Calculate weekly EMA 200 for trend filter
    ema_200_1w = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate daily volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss (optional - can use signal=0 for exit)
    entry_price = np.full(n, np.nan)
    
    for i in range(52, n):  # Start after Senkou B calculation period
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK cross: Tenkan-sen crossing Kijun-sen
        tk_cross_up = (tenkan_aligned[i] > kijun_aligned[i]) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
        tk_cross_down = (tenkan_aligned[i] < kijun_aligned[i]) and (tenkan_aligned[i-1] >= kijun_aligned[i-1])
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_200_1w_aligned[i]
        weekly_downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Volume confirmation: current daily volume > 1.5x 20-period average
        idx_1d = i // 4  # Approximate 6h to 1d index (4x 6h = 1d)
        if idx_1d >= len(daily_volume):
            signals[i] = 0.0
            continue
        volume_confirmed = daily_volume[idx_1d] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Entry conditions
        enter_long = price_above_cloud and tk_cross_up and weekly_uptrend and volume_confirmed
        enter_short = price_below_cloud and tk_cross_down and weekly_downtrend and volume_confirmed
        
        # Exit conditions: TK cross in opposite direction or price re-enters cloud
        exit_long = position == 1 and (tk_cross_down or not price_above_cloud)
        exit_short = position == -1 and (tk_cross_up or not price_below_cloud)
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]  # record entry price at close (filled next bar open)
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]  # record entry price at close (filled next bar open)
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "6h_1w_1d_ichimoku_cloud_trend_volume_v1"
timeframe = "6h"
leverage = 1.0