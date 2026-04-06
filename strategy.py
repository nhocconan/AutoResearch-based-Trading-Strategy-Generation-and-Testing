#!/usr/bin/env python3
"""
6h Ichimoku Cloud + Daily Pivot Points with Volume Confirmation
Hypothesis: Ichimoku provides trend direction (price above/below cloud) and momentum (TK cross),
while daily pivots identify key support/resistance levels. Volume confirmation filters weak breakouts.
Trades only when price breaks above/below cloud with TK cross and volume > 1.5x average.
Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity with fee minimization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_daily_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Ichimoku and pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    def calculate_ichimoku(high, low, close):
        n = len(high)
        tenkan = np.full(n, np.nan)
        kijun = np.full(n, np.nan)
        senkou_a = np.full(n, np.nan)
        senkou_b = np.full(n, np.nan)
        chikou = np.full(n, np.nan)
        
        # Tenkan-sen (9-period)
        for i in range(n):
            if i >= 8:
                high_9 = np.max(high[i-8:i+1])
                low_9 = np.min(low[i-8:i+1])
                tenkan[i] = (high_9 + low_9) / 2
        
        # Kijun-sen (26-period)
        for i in range(n):
            if i >= 25:
                high_26 = np.max(high[i-25:i+1])
                low_26 = np.min(low[i-25:i+1])
                kijun[i] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, shifted 26 periods ahead
        for i in range(n):
            if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
                senkou_a[i] = (tenkan[i] + kijun[i]) / 2
        
        # Senkou Span B (Leading Span B): (52-period high + low)/2, shifted 26 periods ahead
        for i in range(n):
            if i >= 51:
                high_52 = np.max(high[i-51:i+1])
                low_52 = np.min(low[i-51:i+1])
                senkou_b[i] = (high_52 + low_52) / 2
        
        # Chikou Span (Lagging Span): close shifted 26 periods back
        for i in range(n):
            if i >= 26:
                chikou[i] = close[i-26]
        
        return tenkan, kijun, senkou_a, senkou_b, chikou
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    chikou_6h = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    # Daily Pivot Points (standard calculation)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period (max of Ichimoku lookbacks)
    start = max(52, 26)  # For Ichimoku (52-period Senkou B)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Ichimoku signals
        # Price above/below cloud
        above_cloud = close[i] > max(senkou_a_6h[i], senkou_b_6h[i])
        below_cloud = close[i] < min(senkou_a_6h[i], senkou_b_6h[i])
        
        # TK Cross (Tenkan/Kijun cross)
        tk_cross_up = tenkan_6h[i] > kijun_6h[i] and (i == start or tenkan_6h[i-1] <= kijun_6h[i-1])
        tk_cross_down = tenkan_6h[i] < kijun_6h[i] and (i == start or tenkan_6h[i-1] >= kijun_6h[i-1])
        
        # Volume filter (26-period average)
        if i >= 26:
            vol_ma = np.mean(volume[i-26:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below cloud OR TK cross down
            if below_cloud or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above cloud OR TK cross up
            if above_cloud or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Ichimoku signal + volume filter
            # Long: price above cloud + TK cross up + volume
            # Short: price below cloud + TK cross down + volume
            if i >= 26 and above_cloud and tk_cross_up and volume_filter:
                signals[i] = 0.25
                position = 1
            elif i >= 26 and below_cloud and tk_cross_down and volume_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Volume-Weighted Average Price (VWAP) Deviation + Daily ATR Bands
Hypothesis: Price tends to revert to VWAP with oversold/overbought conditions identified 
by deviations beyond 1.5x daily ATR. Uses mean reversion in ranging markets and 
trend continuation when price breaks VWAP with volume confirmation. 
Target: 80-160 total trades over 4 years (20-40/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_vwap_atr_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for ATR calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR (14-period) on daily timeframe
    def calculate_atr(high, low, close, period=14):
        n = len(high)
        atr = np.full(n, np.nan)
        if n < period:
            return atr
        
        # True Range
        tr = np.full(n, np.nan)
        for i in range(n):
            if i == 0:
                tr[i] = high[i] - low[i]
            else:
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # ATR using Wilder's smoothing
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3
    vwap = np.full(n, np.nan)
    cum_vol = np.full(n, np.nan)
    
    for i in range(n):
        if i == 0:
            vwap[i] = typical_price[i]
            cum_vol[i] = volume[i]
        else:
            cum_vol[i] = cum_vol[i-1] + volume[i]
            if cum_vol[i] > 0:
                vwap[i] = (vwap[i-1] * cum_vol[i-1] + typical_price[i] * volume[i]) / cum_vol[i]
            else:
                vwap[i] = vwap[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 14  # For ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr_6h[i]) or np.isnan(vwap[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # VWAP deviation in ATR units
        deviation = (close[i] - vwap[i]) / atr_6h[i] if atr_6h[i] > 0 else 0
        
        # Volume filter (14-period average)
        if i >= 14:
            vol_ma = np.mean(volume[i-14:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits
        if position == 1:  # long position
            # Exit: price returns to VWAP (mean reversion) OR stop loss
            if deviation <= 0.2:  # Returned near VWAP
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to VWAP (mean reversion) OR stop loss
            if deviation >= -0.2:  # Returned near VWAP
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: VWAP deviation + volume filter
            # Long: price < -1.5 * ATR below VWAP (oversold) + volume
            # Short: price > 1.5 * ATR above VWAP (overbought) + volume
            if i >= 14 and deviation < -1.5 and volume_filter:
                signals[i] = 0.25
                position = 1
            elif i >= 14 and deviation > 1.5 and volume_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Weekly Pivot Points + Daily Trend Filter + Volume Confirmation
Hypothesis: Weekly pivot points provide strong support/resistance levels that 
institutional traders watch. Combined with daily EMA trend filter to ensure 
we trade in the direction of the higher timeframe trend, and volume confirmation 
to filter false breakouts. Target: 70-140 total trades over 4 years (17-35/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_daily_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot points (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot Points (standard calculation)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Load daily data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 50-period EMA on daily timeframe
    ema_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        multiplier = 2 / (50 + 1)
        ema_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align daily EMA to 6h timeframe
    ema_1d_6h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 50  # For daily EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_6h[i]) or np.isnan(pivot_6h[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend direction from daily EMA
        daily_uptrend = close[i] > ema_1d_6h[i]
        daily_downtrend = close[i] < ema_1d_6h[i]
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below weekly S1 OR daily trend turns down
            if close[i] < s1_6h[i] or not daily_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above weekly R1 OR daily trend turns up
            if close[i] > r1_6h[i] or not daily_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: weekly pivot bounce + daily trend + volume
            # Long: price > weekly S1 AND price < weekly pivot AND daily uptrend + volume
            # Short: price < weekly R1 AND price > weekly pivot AND daily downtrend + volume
            if i >= 20 and close[i] > s1_6h[i] and close[i] < pivot_6h[i] and daily_uptrend and volume_filter:
                signals[i] = 0.25
                position = 1
            elif i >= 20 and close[i] < r1_6h[i] and close[i] > pivot_6h[i] and daily_downtrend and volume_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Volume-Weighted RSI (VRSI) + Daily ADX Trend Filter
Hypothesis: VRSI (RSI calculated using volume-weighted price) identifies 
overextended conditions, while daily ADX filters for trending vs ranging markets.
In trending markets (ADX > 25), we follow VRSI extremes; in ranging markets 
(ADX < 20), we mean revert at VRSI extremes. Target: 60-120 total trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_vrsi_daily_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for ADX calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on daily timeframe
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period * 2:
            return np.full(n, np.nan)
        
        # True Range and Directional Movement
        tr = np.full(n, np.nan)
        dm_plus = np.full(n, np.nan)
        dm_minus = np.full(n, np.nan)
        
        for i in range(n):
            if i == 0:
                tr[i] = high[i] - low[i]
                dm_plus[i] = 0
                dm_minus[i] = 0
            else:
                high_diff = high[i] - high[i-1]
                low_diff = low[i-1] - low[i]
                
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                
                if high_diff > low_diff and high_diff > 0:
                    dm_plus[i] = high_diff
                else:
                    dm_plus[i] = 0
                    
                if low_diff > high_diff and low_diff > 0:
                    dm_minus[i] = low_diff
                else:
                    dm_minus[i] = 0
        
        # Smoothed values
        atr = np.full(n, np.nan)
        s_dm_plus = np.full(n, np.nan)
        s_dm_minus = np.full(n, np.nan)
        
        # Initial averages
        if n >= period:
            atr[period-1] = np.mean(tr[:period])
            s_dm_plus[period-1] = np.mean(dm_plus[:period])
            s_dm_minus[period-1] = np.mean(dm_minus[:period])
            
            # Wilder's smoothing
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                s_dm_plus[i] = (s_dm_plus[i-1] * (period-1) + dm_plus[i]) / period
                s_dm_minus[i] = (s_dm_minus[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.full(n, np.nan)
        di_minus = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        for i in range(period-1, n):
            if atr[i] > 0:
                di_plus[i] = (s_dm_plus[i] / atr[i]) * 100
                di_minus[i] = (s_dm_minus[i] / atr[i]) * 100
                dx[i] = (abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
        
        # ADX (smoothed DX)
        adx = np.full(n, np.nan)
        if n >= period * 2:
            adx[2*period-1] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_6h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Volume-Weighted RSI (VRSI)
    # Typical price weighted by volume
    typical_price = (high + low + close) / 3
    vwp = typical_price * volume  # Volume-weighted price
    
    # Calculate changes in VWP
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = vwp[i] - vwp[i-1]
    
    # Separate gains and losses
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Calculate average gain and loss (Wilder's smoothing)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    period = 14
    
    if n >= period:
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        
        for i in range(period, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    # Calculate RSI
    rs = np.full(n, np.nan)
    rsi = np.full(n, 50.0)  # Default to neutral
    
    for i in range(period-1, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(period * 2, 14)  # For ADX and VRSI
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(adx_6h[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter (14-period average)
        if i >= 14:
            vol_ma = np.mean(volume[i-14:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits
        if position == 1:  # long position
            # Exit: RSI returns to 50 OR ADX drops below 20 (trend weakening)
            if rsi[i] >= 50 or adx_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI returns to 50 OR ADX drops below 20 (trend weakening)
            if rsi[i] <= 50 or adx_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on regime
            # Trending market (ADX > 25): follow momentum
            # Ranging market (ADX < 20): mean revert
            if adx_6h[i] > 25:  # Trending
                # Long: RSI < 30 (oversold) + volume
                # Short: RSI > 70 (overbought) + volume
                if i >= 14 and rsi[i] < 30 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                elif i >= 14 and rsi[i] > 70 and volume_filter:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif adx_6h[i] < 20:  # Ranging
                # Long: RSI < 30 (oversold) + volume
                # Short: RSI > 70 (overbought) + volume
                if i >= 14 and rsi[i] < 30 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                elif i >= 14 and rsi[i] > 70 and volume_filter:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # Transition zone (20 <= ADX <= 25)
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Kaufman's Adaptive Moving Average (KAMA) + Daily Bollinger Bands Width
Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets.
Bollinger Bands Width identifies regime: low BBW = ranging (mean revert at KAMA),
high BBW = trending (follow KAMA crossovers). Target: 50-100 total trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_kama_bbw_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Bollinger Bands Width (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands and Width (20, 2)
    def calculate_bbw(close, period=20, std_dev=2):
        n = len(close)
        if n < period:
            return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        
        # Moving average
        ma = np.full(n, np.nan)
        for i in range(n):
            if i >= period - 1:
                ma[i] = np.mean(close[i-period+1:i+1])
        
        # Standard deviation
        std = np.full(n, np.nan)
        for i in range(n):
            if i >= period - 1:
                std[i] = np.std(close[i-period+1:i+1])
        
        # Upper and lower bands
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        width = (upper - lower) / ma  # Normalized width
        
        return upper, lower, width
    
    _, _, bbw_1d = calculate_bbw(close_1d, 20, 2)
    bbw_6h = align_htf_to_ltf(prices, df_1d, bbw_1d)
    
    # Price data
    close = prices['close'].values
    
    # Calculate Kaufman's Adaptive Moving Average (KAMA)
    def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
        n = len(close)
        kama = np.full(n, np.nan)
        if n < er_period:
            return kama
        
        # Efficiency Ratio
        change = np.abs(np.d