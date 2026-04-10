#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + 1d Weekly Pivot Confluence for 6h timeframe
# - Primary: 6h Ichimoku TK Cross (Tenkan/Kijun) with price above/below cloud for trend confirmation
# - HTF: 1d Weekly Pivot levels (PP, R1/S1, R2/S2) for institutional reference points
# - Entry: Long when TK Cross bullish + price > Kumo cloud + price > Weekly PP + volume > 1.2x MA
#          Short when TK Cross bearish + price < Kumo cloud + price < Weekly PP + volume > 1.2x MA
# - Exit: Opposite TK Cross or price crosses Weekly Pivot levels
# - Session: 08-20 UTC for liquidity
# - Position: 0.25 discrete to minimize fee churn
# - Target: 80-120 total trades over 4 years (20-30/year) for 6h timeframe
# - Works in bull/bear: Ichimoku adapts to trends, Weekly Pivots provide mean reversion levels in ranges

name = "6h_1d_ichimoku_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Kumo cloud boundaries (shifted 26 periods ahead)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Calculate 1d Weekly Pivot Points (using prior week's OHLC)
    # For daily data, weekly pivot uses prior week's high, low, close
    # We'll approximate using rolling window on daily data
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values  # 5 trading days
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values  # approx
    
    # Weekly Pivot Point (PP) = (High + Low + Close) / 3
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3
    
    # Weekly Resistance 1 (R1) = (2 * PP) - Low
    weekly_r1 = 2 * weekly_pp - weekly_low
    
    # Weekly Support 1 (S1) = (2 * PP) - High
    weekly_s1 = 2 * weekly_pp - weekly_high
    
    # Weekly Resistance 2 (R2) = PP + (High - Low)
    weekly_r2 = weekly_pp + (weekly_high - weekly_low)
    
    # Weekly Support 2 (S2) = PP - (High - Low)
    weekly_s2 = weekly_pp - (weekly_high - weekly_low)
    
    # Align HTF Ichimoku and Weekly Pivot data to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    senkou_a_shifted_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_shifted)
    senkou_b_shifted_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_shifted)
    
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    # Calculate 1d volume MA(20) for confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid or outside session
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(senkou_a_shifted_aligned[i]) or np.isnan(senkou_b_shifted_aligned[i]) or
            np.isnan(weekly_pp_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.2x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.2 * volume_ma_20_1d_aligned[i]
        
        # Determine Ichimoku trend
        # Bullish: Tenkan > Kijun AND price > Senkou Span A AND price > Senkou Span B
        # Bearish: Tenkan < Kijun AND price < Senkou Span A AND price < Senkou Span B
        price = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        
        ichimoku_bullish = (tenkan_val > kijun_val) and (price > senkou_a_val) and (price > senkou_b_val)
        ichimoku_bearish = (tenkan_val < kijun_val) and (price < senkou_a_val) and (price < senkou_b_val)
        
        # Weekly Pivot levels
        pp = weekly_pp_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        r2 = weekly_r2_aligned[i]
        s2 = weekly_s2_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Ichimoku bullish + price > Weekly PP + volume confirmation
            if ichimoku_bullish and price > pp and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Ichimoku bearish + price < Weekly PP + volume confirmation
            elif ichimoku_bearish and price < pp and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Opposite Ichimoku Cross or price crosses Weekly Pivot levels
            if position == 1:  # Long position
                exit_condition = (not ichimoku_bullish) or (price < s1)  # Exit if bearish or breaks S1
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (not ichimoku_bearish) or (price > r1)  # Exit if bullish or breaks R1
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals