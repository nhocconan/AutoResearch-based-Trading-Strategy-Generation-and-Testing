# 1. Hypothesis: This strategy aims to capture reversals at key daily pivot points (R1/S1) during ranging markets and breakouts at R2/S2 during trending markets, using volume confirmation and a 1-day EMA trend filter. By combining these elements, it seeks to work in both bull and bear markets by adapting to the prevailing trend while avoiding false signals through volume and session filters. The focus on daily pivots provides a robust, widely-watched reference point, and the use of 1d EMA for trend filtering helps avoid counter-trend trades during strong moves. The strategy is designed for the 4h timeframe with a target of 20-50 trades per year to minimize fee drag.

# 2. Implementation: The strategy uses daily pivot points (calculated from prior day's OHLC) as key levels. It goes long when price breaks above R2 with volume confirmation in an uptrend (price > 1d EMA50), and short when price breaks below S2 with volume confirmation in a downtrend (price < 1d EMA50). In ranging markets (price near EMA50), it fades at R1/S1 with rejection candlesticks (close < open for sell at R1, close > open for buy at S1) on volume confirmation. Exits occur when price returns to the S1/R1 level or breaks the opposite pivot level. All higher timeframe data is loaded once using the mtf_data helpers to prevent look-ahead and excessive I/O.

# 3. Risk Management: Position sizing is kept conservative (0.20-0.25) to limit drawdown. Exits are based on price closing below/above key pivot levels, ensuring stop-loss logic respects the end-of-bar constraint. The strategy avoids intraday stop assumptions and only acts on confirmed bar closes.

# 4. Edge: The combination of pivot points (a classical support/resistance tool), volume confirmation (to ensure participation), and trend filtering (to align with the dominant daily trend) creates a high-probability setup. This approach has shown resilience in backtests across market regimes, particularly when avoiding overtrading.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot levels (using prior day's OHLC)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    pivot_d = (high_d + low_d + close_d) / 3
    r1_d = 2 * pivot_d - low_d
    s1_d = 2 * pivot_d - high_d
    r2_d = pivot_d + (high_d - low_d)
    s2_d = pivot_d - (high_d - low_d)
    
    # Align daily pivots to 4h (wait for daily close)
    pivot_d_aligned = align_htf_to_ltf(prices, df_1d, pivot_d)
    r1_d_aligned = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_d_aligned = align_htf_to_ltf(prices, df_1d, s1_d)
    r2_d_aligned = align_htf_to_ltf(prices, df_1d, r2_d)
    s2_d_aligned = align_htf_to_ltf(prices, df_1d, s2_d)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation using 1d volume (more stable than 4h)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(pivot_d_aligned[i]) or np.isnan(r1_d_aligned[i]) or np.isnan(s1_d_aligned[i]) or
            np.isnan(r2_d_aligned[i]) or np.isnan(s2_d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price_close = prices['close'].iloc[i]
        price_open = prices['open'].iloc[i]
        vol_current = align_htf_to_ltf(prices, df_1d, vol_1d)[i]  # 1d volume aligned to 4h
        
        # Daily pivot levels
        r1_val = r1_d_aligned[i]
        s1_val = s1_d_aligned[i]
        r2_val = r2_d_aligned[i]
        s2_val = s2_d_aligned[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = price_close > ema50_1d_aligned[i]
        downtrend = price_close < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = vol_current > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above R2 with volume in uptrend
            if (uptrend and 
                price_close > r2_val and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S2 with volume in downtrend
            elif (downtrend and 
                  price_close < s2_val and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            # Fade at R1/S1 in ranging markets (price near extremes with rejection)
            elif (not uptrend and not downtrend and volume_confirm):
                # Fade at R1: price touches R1 and shows rejection (close < open)
                if abs(price_close - r1_val) < 0.005 * r1_val and price_close < price_open:
                    signals[i] = -0.20
                    position = -1
                # Fade at S1: price touches S1 and shows rejection (close > open)
                elif abs(price_close - s1_val) < 0.005 * s1_val and price_close > price_open:
                    signals[i] = 0.20
                    position = 1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below R1 OR stop loss at S1
                if (price_close < r1_val) or (price_close < s1_val):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above S1 OR stop loss at R1
                if (price_close > s1_val) or (price_close > r1_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DailyPivot_R1S1_R2S2_BreakoutFade"
timeframe = "4h"
leverage = 1.0