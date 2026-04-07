#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day volume confirmation and weekly trend filter
# Long when price breaks above Donchian(20) high + volume > 1.5x 20-period average + weekly close > weekly SMA(50)
# Short when price breaks below Donchian(20) low + volume > 1.5x 20-period average + weekly close < weekly SMA(50)
# Exit when price crosses Donchian midpoint or volume drops below average
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day volume for confirmation and 1-week SMA for trend filter
# Target: 100-200 total trades over 4 years (25-50/year)

name = "4h_donchian20_1d_vol_1w_trend_v1"
timeframe = "4h"
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
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week SMA(50)
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    sma_50 = close_1w_s.rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50)
    
    # 4-hour Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2
    
    # 4-period ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(sma_50_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midpoint or volume drops below average
            elif close[i] < donchian_mid[i] or volume[i] < volume_ma_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midpoint or volume drops below average
            elif close[i] > donchian_mid[i] or volume[i] < volume_ma_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and trend filter
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            
            # Long: price breaks above Donchian high + volume confirmation + weekly close > SMA(50)
            if close[i] > high_max[i] and volume_filter and close_1w[-1] > sma_50[-1] if len(close_1w) > 0 and len(sma_50) > 0 else False:
                # Need to get the aligned weekly close and SMA for current bar
                # Since we can't access future weekly data, we use the aligned values
                # We'll check if the aligned weekly close > aligned SMA(50)
                # But we don't have weekly close aligned, so we'll use a proxy: if current 4h close > weekly SMA
                # This is approximate but avoids look-ahead
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume confirmation + weekly close < SMA(50)
            elif close[i] < low_min[i] and volume_filter and close_1w[-1] < sma_50[-1] if len(close_1w) > 0 and len(sma_50) > 0 else False:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

Note: The weekly trend filter implementation has an issue - we need to properly align the weekly close price. Let me fix this.

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day volume confirmation and weekly trend filter
# Long when price breaks above Donchian(20) high + volume > 1.5x 20-period average + weekly close > weekly SMA(50)
# Short when price breaks below Donchian(20) low + volume > 1.5x 20-period average + weekly close < weekly SMA(50)
# Exit when price crosses Donchian midpoint or volume drops below average
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day volume for confirmation and 1-week SMA for trend filter
# Target: 100-200 total trades over 4 years (25-50/year)

name = "4h_donchian20_1d_vol_1w_trend_v1"
timeframe = "4h"
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
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week SMA(50) and align weekly close
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    sma_50 = close_1w_s.rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # 4-hour Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2
    
    # 4-period ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(sma_50_aligned[i]) or 
            np.isnan(close_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midpoint or volume drops below average
            elif close[i] < donchian_mid[i] or volume[i] < volume_ma_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midpoint or volume drops below average
            elif close[i] > donchian_mid[i] or volume[i] < volume_ma_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and trend filter
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            
            # Long: price breaks above Donchian high + volume confirmation + weekly close > SMA(50)
            if close[i] > high_max[i] and volume_filter and close_1w_aligned[i] > sma_50_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume confirmation + weekly close < SMA(50)
            elif close[i] < low_min[i] and volume_filter and close_1w_aligned[i] < sma_50_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals