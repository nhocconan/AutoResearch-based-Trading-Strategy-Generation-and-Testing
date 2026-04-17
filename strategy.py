#/usr/bin/env python3
"""
6h_1D_Weekly_Pivot_Reversion
Hypothesis: Price tends to revert to weekly pivot points (calculated from Monday's OHLC) on 6h timeframe.
In ranging markets (identified by 1d ADX < 25), we take mean-reversion trades at weekly S1/R1.
In trending markets (1d ADX >= 25), we take breakout trades at weekly S2/R2.
This adapts to both bull and bear regimes by using ADX to filter market state.
Weekly pivots provide strong institutional levels that price respects.
Position size: ±0.25. Max 0.25 to control drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX and weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX(14) for regime detection
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smooth TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
        def wilders_smooth(data, period):
            result = np.zeros_like(data)
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        atr = wilders_smooth(tr, period)
        dm_plus_smooth = wilders_smooth(dm_plus, period)
        dm_minus_smooth = wilders_smooth(dm_minus, period)
        
        # DI+ and DI-
        di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
        di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = wilders_smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate weekly pivots from Monday's OHLC
    # We'll calculate pivots for each week and align them to 6h bars
    weekly_high = np.full(n, np.nan)
    weekly_low = np.full(n, np.nan)
    weekly_close = np.full(n, np.nan)
    
    # Find Monday start of each week (0 = Monday in pandas)
    # We'll use the date from open_time to determine week start
    dates = pd.to_datetime(prices['open_time'])
    week_start = dates.dt.weekday == 0  # Monday
    
    # For each bar, if it's Monday, capture the week's OHLC
    current_week_high = np.full(n, np.nan)
    current_week_low = np.full(n, np.nan)
    current_week_close = np.full(n, np.nan)
    
    week_high = np.nan
    week_low = np.nan
    week_close = np.nan
    in_week = False
    
    for i in range(n):
        if week_start.iloc[i] and not in_week:
            # Start of new week
            week_high = high[i]
            week_low = low[i]
            week_close = close[i]
            in_week = True
        elif in_week:
            # Update week's OHLC
            week_high = max(week_high, high[i])
            week_low = min(week_low, low[i])
            week_close = close[i]
        
        # Store current week's values for all bars in the week
        if in_week:
            current_week_high[i] = week_high
            current_week_low[i] = week_low
            current_week_close[i] = week_close
        
        # Reset if we've passed Sunday (6 = Sunday)
        if week_start.iloc[i] and in_week and i > 0:
            in_week = False
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    weekly_pivot = (current_week_high + current_week_low + current_week_close) / 3.0
    
    # Support and Resistance levels
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    weekly_range = current_week_high - current_week_low
    weekly_r1 = 2 * weekly_pivot - current_week_low
    weekly_s1 = 2 * weekly_pivot - current_week_high
    weekly_r2 = weekly_pivot + weekly_range
    weekly_s2 = weekly_pivot - weekly_range
    
    # Align weekly pivot levels to 6h (they change only on Monday)
    weekly_pivot_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(n)}), weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(n)}), weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(n)}), weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(n)}), weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(n)}), weekly_s2)
    
    # Volume filter: current volume > 1.2x 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 14)  # Volume MA20, ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(weekly_r2_aligned[i]) or 
            np.isnan(weekly_s2_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.2 * volume_ma20[i])
        
        # Determine market regime: ADX < 25 = ranging, ADX >= 25 = trending
        ranging_market = adx_1d_aligned[i] < 25
        trending_market = adx_1d_aligned[i] >= 25
        
        # Price levels
        price = close[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        r2 = weekly_r2_aligned[i]
        s2 = weekly_s2_aligned[i]
        pivot = weekly_pivot_aligned[i]
        
        if position == 0:
            # In ranging market: mean reversion at S1/R1
            if ranging_market and volume_filter:
                # Long near S1 with rejection (close > open and near S1)
                if price <= s1 * 1.005 and close[i] > open_price[i]:  # Within 0.5% of S1 and bullish candle
                    signals[i] = 0.25
                    position = 1
                # Short near R1 with rejection (close < open and near R1)
                elif price >= r1 * 0.995 and close[i] < open_price[i]:  # Within 0.5% of R1 and bearish candle
                    signals[i] = -0.25
                    position = -1
            
            # In trending market: breakout at S2/R2
            elif trending_market and volume_filter:
                # Long breakout above R2
                if price >= r2 * 0.995:  # Within 0.5% above R2
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below S2
                elif price <= s2 * 1.005:  # Within 0.5% below S2
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price reaches pivot or opposite signal
            if price >= pivot * 0.995:  # Near pivot
                signals[i] = 0.0
                position = 0
            elif price <= s1 * 1.005 and close[i] < open_price[i]:  # Rejection at S1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches pivot or opposite signal
            if price <= pivot * 1.005:  # Near pivot
                signals[i] = 0.0
                position = 0
            elif price >= r1 * 0.995 and close[i] > open_price[i]:  # Rejection at R1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1D_Weekly_Pivot_Reversion"
timeframe = "6h"
leverage = 1.0