#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian channel breakout with weekly EMA trend filter and daily volume confirmation
# Long when price breaks above daily Donchian(20) high + volume > 1.5x daily average + weekly EMA20 uptrend
# Short when price breaks below daily Donchian(20) low + volume > 1.5x daily average + weekly EMA20 downtrend
# Exit when price crosses Donchian midline or volume drops below daily average
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses daily volume for confirmation and weekly EMA20 for trend direction
# Target: 50-100 total trades over 4 years (12.5-25/year)

name = "1d_donchian20_vol_1w_ema_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_1d_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_ma)
    
    # Calculate weekly EMA20 for trend
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema20_1w = close_1w_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Daily Donchian channel (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_1d_ma_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or 
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
            # Exit: price crosses Donchian midline or volume drops below average
            elif close[i] <= donchian_mid[i] or volume[i] < volume_1d_ma_aligned[i]:
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
            # Exit: price crosses Donchian midline or volume drops below average
            elif close[i] >= donchian_mid[i] or volume[i] < volume_1d_ma_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and trend filter
            # Volume spike: > 1.5x average daily volume
            volume_confirmed = volume[i] > 1.5 * volume_1d_ma_aligned[i]
            # Trend filter: price above/below weekly EMA20
            uptrend = close[i] > ema20_1w_aligned[i]
            downtrend = close[i] < ema20_1w_aligned[i]
            
            # Long: break above Donchian high, volume confirmed, uptrend
            if close[i] > donchian_high[i] and volume_confirmed and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below Donchian low, volume confirmed, downtrend
            elif close[i] < donchian_low[i] and volume_confirmed and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian channel breakout with weekly EMA trend filter and daily volume confirmation
# Long when price breaks above daily Donchian(20) high + volume > 1.5x daily average + weekly EMA20 uptrend
# Short when price breaks below daily Donchian(20) low + volume > 1.5x daily average + weekly EMA20 downtrend
# Exit when price crosses Donchian midline or volume drops below daily average
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses daily volume for confirmation and weekly EMA20 for trend direction
# Target: 50-100 total trades over 4 years (12.5-25/year)

name = "1d_donchian20_vol_1w_ema_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_1d_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_ma)
    
    # Calculate weekly EMA20 for trend
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema20_1w = close_1w_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Daily Donchian channel (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_1d_ma_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or 
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
            # Exit: price crosses Donchian midline or volume drops below average
            elif close[i] <= donchian_mid[i] or volume[i] < volume_1d_ma_aligned[i]:
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
            # Exit: price crosses Donchian midline or volume drops below average
            elif close[i] >= donchian_mid[i] or volume[i] < volume_1d_ma_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and trend filter
            # Volume spike: > 1.5x average daily volume
            volume_confirmed = volume[i] > 1.5 * volume_1d_ma_aligned[i]
            # Trend filter: price above/below weekly EMA20
            uptrend = close[i] > ema20_1w_aligned[i]
            downtrend = close[i] < ema20_1w_aligned[i]
            
            # Long: break above Donchian high, volume confirmed, uptrend
            if close[i] > donchian_high[i] and volume_confirmed and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below Donchian low, volume confirmed, downtrend
            elif close[i] < donchian_low[i] and volume_confirmed and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals