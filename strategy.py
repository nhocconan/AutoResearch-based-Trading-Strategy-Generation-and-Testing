#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray index with 1d trend filter for trend-following in bull/bear markets.
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13).
# Long when Bull Power > 0 and Bear Power improving (less negative) with 1d EMA50 uptrend.
# Short when Bear Power < 0 and Bull Power deteriorating (less positive) with 1d EMA50 downtrend.
# Uses volume confirmation to avoid false signals. Target: 75-200 total trades over 4 years.

name = "6h_elderray_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Elder Ray components: EMA13 of close
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Smooth the power signals to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, min_periods=5, adjust=False).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, min_periods=5, adjust=False).mean().values
    
    # Volume filter: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bear Power turns negative or 1d EMA turns down
            elif bear_power_smooth[i] < 0 or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bull Power turns positive or 1d EMA turns up
            elif bull_power_smooth[i] > 0 or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if vol_filter[i]:
                # Long: Bull Power positive AND Bear Power improving (less negative)
                if bull_power_smooth[i] > 0 and bear_power_smooth[i] > bear_power_smooth[i-1]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Bear Power negative AND Bull Power deteriorating (less positive)
                elif bear_power_smooth[i] < 0 and bull_power_smooth[i] < bull_power_smooth[i-1]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Average Price (VWAP) deviation with 1d trend filter.
# Goes long when price closes above VWAP with rising volume and 1d UPTREND.
# Goes short when price closes below VWAP with rising volume and 1d DOWNTREND.
# Uses VWAP deviation bands (1.5 * ATR) for entry and 2 * ATR for stoploss.
# Works in both bull/bear markets by following higher timeframe trend.
# Target: 100-250 total trades over 4 years (25-60/year).

name = "6h_vwap_dev_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # VWAP calculation (session-based, reset daily)
    # Typical price * volume cumulative / volume cumulative
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    # Avoid division by zero
    vwap_den = np.where(vwap_den == 0, 1, vwap_den)
    vwap = vwap_num / vwap_den
    
    # Reset VWAP at daily boundaries (using date change in index)
    # Since we don't have easy access to date in values, we'll use a rolling window
    # Alternative: use 24 periods (4h * 6 = 24) for daily reset approximation
    vwap_reset = pd.Series(vwap).rolling(window=24, min_periods=1).mean().values
    
    # VWAP deviation
    vwap_dev = close - vwap_reset
    
    # ATR for bands and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: rising volume (current > previous)
    vol_rising = volume > np.roll(volume, 1)
    vol_rising[0] = False
    
    # VWAP bands: 1.5 * ATR above/below VWAP
    vwap_upper = vwap_reset + 1.5 * atr
    vwap_lower = vwap_reset - 1.5 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vwap_dev[i]) or 
            np.isnan(vwap_upper[i]) or np.isnan(vwap_lower[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price closes below VWAP or 1d trend turns down
            elif close[i] < vwap_reset[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price closes above VWAP or 1d trend turns up
            elif close[i] > vwap_reset[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price outside VWAP bands with volume confirmation
            if vol_rising[i]:
                # Long: price closes above VWAP upper band AND 1d UPTREND
                if close[i] > vwap_upper[i] and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price closes below VWAP lower band AND 1d DOWNTREND
                elif close[i] < vwap_lower[i] and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_ftf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with 1d volume confirmation and 1w trend filter.
# Goes long when price breaks above 20-period Donchian HIGH with volume > 1.5x average.
# Goes short when price breaks below 20-period Donchian LOW with volume > 1.5x average.
# Uses 1-week EMA50 as trend filter: only take longs in uptrend, shorts in downtrend.
# Includes volatility filter: only trade when ATR(14) > ATR(50) * 0.5 (avoid choppy markets).
# Target: 80-200 total trades over 4 years (20-50/year).

name = "6h_donchian20_1d_vol_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Donchian channels (20-period)
    # Highest high and lowest low over past 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filters from 1d data
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    vol_1d_current = align_htf_to_ltf(prices, df_1d, vol_1d)
    vol_strong = vol_1d_current > (vol_ma_1d_aligned * 1.5)  # 1.5x average daily volume
    
    # Volatility filter: trade only when volatility is elevated
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_14 > (atr_50 * 0.5)  # Avoid low volatility/choppy markets
    
    # ATR for stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_strong[i]) or 
            np.isnan(vol_filter[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian LOW or 1w trend turns down
            elif close[i] < lowest_low[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian HIGH or 1w trend turns up
            elif close[i] > highest_high[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakouts with volume and volatility confirmation
            if vol_strong[i] and vol_filter[i]:
                # Long breakout: price breaks above Donchian HIGH
                if close[i] > highest_high[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below Donchian LOW
                elif close[i] < lowest_low[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Connors RSI (CRSI) with 1d trend filter for mean reversion in trending markets.
# CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
# Long when CRSI < 15 and price > 200-day EMA (uptrend pullback).
# Short when CRSI > 85 and price < 200-day EMA (downtrend rally).
# Uses 1-day EMA50 as trend filter to align with intermediate trend.
# Includes volume confirmation: volume > 1.2x average to avoid low-volume false signals.
# Target: 90-220 total trades over 4 years (22-55/year).

name = "6h_crsi_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # RSI(3) calculation
    def calculate_rsi(prices, period):
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_3 = calculate_rsi(close, 3)
    
    # RSI Streak(2): 2-period RSI of consecutive up/down days
    up_days = np.where(close > np.roll(close, 1), 1, 0)
    down_days = np.where(close < np.roll(close, 1), 1, 0)
    up_streak = pd.Series(up_days).rolling(window=2, min_periods=2).sum().values
    down_streak = pd.Series(down_days).rolling(window=2, min_periods=2).sum().values
    streak_raw = up_streak - down_streak  # Positive for up streaks, negative for down
    # Convert to 0-100 scale: map -2,-1,0,1,2 to 0,25,50,75,100
    rsi_streak = 50 + (streak_raw * 12.5)
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Percent Rank(100): where current price ranks in last 100 periods
    def calculate_percent_rank(arr, window):
        rank = np.zeros_like(arr)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            window_data = arr[start:i+1]
            if len(window_data) > 0:
                rank[i] = (np.sum(window_data <= arr[i]) / len(window_data)) * 100
            else:
                rank[i] = 50
        return rank
    
    percent_rank = calculate_percent_rank(close, 100)
    
    # Connors RSI
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    
    # Volume filter: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.2)  # 20% above average volume
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(crsi[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_filter[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: CRSI > 70 (overbought) or trend turns down
            elif crsi[i] > 70 or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: CRSI < 30 (oversold) or trend turns up
            elif crsi[i] < 30 or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for mean reversion entries
            if vol_filter[i]:
                # Long: CRSI oversold (<15) in uptrend (price > 1d EMA50)
                if crsi[i] < 15 and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: CRSI overbought (>85) in downtrend (price < 1d EMA50)
                elif crsi[i] > 85 and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Adaptive Supertrend with 1d volatility regime filter.
# Uses ATR multiplier that adapts to market volatility: higher multiplier in high vol, lower in low vol.
# Long when price > Supertrend(UPPER) and 1d ADX > 25 (trending market).
# Short when price < Supertrend(LOWER) and 1d ADX > 25.
# Includes volatility filter: only trade when ATR(14) > ATR(50) * 0.6 to avoid chop.
# Adaptive multiplier: base=3.0, scales with ATR ratio (ATR14/ATR50).
# Target: 70-180 total trades over 4 years (17-45/year).

name = "6h_adaptivesupertrend_1d_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ADX and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        up_sum = pd.Series(up_move).rolling(window=period, min_periods=period).sum().values
        down_sum = pd.Series(down_move).rolling(window=period, min_periods=period).sum().values
        
        # Directional Indicators
        plus_di = 100 * (up_sum / (tr_sum + 1e-10))
        minus_di = 100 * (down_sum / (tr_sum + 1e-10))
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # ATR for Supertrend and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_14 > (atr_50 * 0.6)  # Avoid low volatility/chop
    
    # Adaptive Supertrend calculation
    def supertrend(high, low, close, atr, period=10):
        # Basic upper and lower bands
        hl2 = (high + low) / 2.0
        upper_band = hl2 + (atr * 3.0)  # Will be made adaptive
        lower_band = hl2 - (atr * 3.0)
        
        # Initialize arrays
        final_upper = np.zeros(len(close))
        final_lower = np.zeros(len(close))
        supertrend = np.zeros(len(close))
        trend = np.zeros(len(close))  # 1 for uptrend, -1 for downtrend
        
        # Adaptive multiplier based on ATR ratio
        atr_ratio = atr_14 / (atr_50 + 1e-10)
        multiplier = 3.0 * (0.5 + atr_ratio)  # Scales from 1.5x to 4.5x
        
        # Recalculate bands with adaptive multiplier
        upper_band = hl2 + (atr * multiplier)
        lower_band = hl2 - (atr * multiplier)
        
        for i in range(1, len(close)):
            # Upper band logic
            if close[i-1] > final_upper[i-1]:
                final_upper[i] = upper_band[i]
            else:
                final_upper[i] = min(upper_band[i], final_upper[i-1])
            
            # Lower band logic
            if close[i-1] < final_lower[i-1]:
                final_lower[i] = lower_band[i]
            else:
                final_lower[i] = max(lower_band[i], final_lower[i-1])
            
            # Trend logic
            if close[i] > final_upper[i-1]:
                trend[i] = 1
            elif close[i] < final_lower[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
                if trend[i] == -1 and final_upper[i] < final_upper[i-1]:
                    final_upper[i] = final_upper[i-1]
                if trend[i] == 1 and final_lower[i] > final_lower[i-1]:
                    final_lower[i] = final_lower[i-1]
            
            # Supertrend value
            if trend[i] == 1:
                supertrend[i] = final_lower[i]
            else:
                supertrend[i] = final_upper[i]
        
        return supertrend, trend
    
    # Calculate Supertrend
    st, st_trend = supertrend(high, low, close, atr_14, 10)
    
    # Volume filter: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(st[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_filter[i]) or np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price closes below Supertrend or ADX < 25 (losing trend)
            elif close[i] < st[i] or adx_1d_al