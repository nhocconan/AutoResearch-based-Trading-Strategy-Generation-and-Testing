#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# Long when price breaks above 20-period Donchian high, 1d close > 1d EMA50, volume > 1.5x 4h volume average
# Short when price breaks below 20-period Donchian low, 1d close < 1d EMA50, volume > 1.5x 4h volume average
# Exit when price returns to Donchian midpoint (mean of 20-period high/low) or trend reverses
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 80-150 total trades over 4 years (20-38/year)

name = "4h_donchian20_1d_ema50_vol_v1"
timeframe = "4h"
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
    
    # 4h Donchian(20) - high and low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h volume average for confirmation
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma[i]) or 
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
            # Exit: price returns to Donchian midpoint or trend reverses
            elif close[i] < donchian_mid[i] or close[i] < ema50_1d_aligned[i]:
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
            # Exit: price returns to Donchian midpoint or trend reverses
            elif close[i] > donchian_mid[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Donchian breakout, trend alignment, and volume confirmation
            # Bullish breakout: price breaks above Donchian high
            bullish_breakout = close[i] > donchian_high[i]
            # Bearish breakout: price breaks below Donchian low
            bearish_breakout = close[i] < donchian_low[i]
            
            # Long: bullish breakout, 1d uptrend, volume spike
            if (bullish_breakout and
                close[i] > ema50_1d_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish breakout, 1d downtrend, volume spike
            elif (bearish_breakout and
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d with volume spike and Choppiness regime filter
# Long when price touches Camarilla L3 support, 1d EMA50 uptrend, volume > 2x 4h avg, Choppiness > 61.8 (range)
# Short when price touches Camarilla H3 resistance, 1d EMA50 downtrend, volume > 2x 4h avg, Choppiness > 61.8
# Exit when price reaches Camarilla H4/L4 or trend reverses
# Stoploss at 2.0 * ATR(14)
# Position size: 0.30 (30% of capital)
# Target: 60-120 total trades over 4 years (15-30/year)

name = "4h_camarilla_1d_vol_chop_v1"
timeframe = "4h"
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
    
    # 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels calculation
    range_ = prev_high - prev_low
    camarilla_h4 = prev_close + 1.1 * range_ * 1.1 / 2
    camarilla_h3 = prev_close + 1.1 * range_ * 1.1 / 4
    camarilla_l3 = prev_close - 1.1 * range_ * 1.1 / 4
    camarilla_l4 = prev_close - 1.1 * range_ * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1d trend filter (EMA50)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h volume average for confirmation
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14) for regime filter
    atr1 = high - low
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    atr2[0] = atr1[0]
    atr3[0] = atr1[0]
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(chop[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches H4/L4 or trend reverses
            elif close[i] >= h4_aligned[i] or close[i] <= l4_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches H4/L4 or trend reverses
            elif close[i] <= l4_aligned[i] or close[i] >= h4_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Look for entries at Camarilla levels with volume spike and range regime
            # Near L3 support (within 0.1% tolerance)
            near_l3 = abs(close[i] - l3_aligned[i]) / l3_aligned[i] < 0.001
            # Near H3 resistance (within 0.1% tolerance)
            near_h3 = abs(close[i] - h3_aligned[i]) / h3_aligned[i] < 0.001
            
            # Long: near L3, 1d uptrend, volume spike, range regime (Chop > 61.8)
            if (near_l3 and
                close[i] > ema50_1d_aligned[i] and
                volume[i] > 2.0 * volume_ma[i] and
                chop[i] > 61.8):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: near H3, 1d downtrend, volume spike, range regime (Chop > 61.8)
            elif (near_h3 and
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > 2.0 * volume_ma[i] and
                  chop[i] > 61.8):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h CRSI (Connors RSI) with Choppiness regime filter and Donchian exit
# Long when CRSI < 15, price > 4h EMA200, Choppiness > 61.8 (range)
# Short when CRSI > 85, price < 4h EMA200, Choppiness > 61.8 (range)
# Exit when price reaches Donchian(20) midpoint or trend reverses
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 70-140 total trades over 4 years (18-35/year)

name = "4h_crsi_chop_donchian_exit_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA200 for trend filter
    close_series = pd.Series(close)
    ema200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 4h RSI(3) for CRSI
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(span=3, adjust=False, min_periods=3).mean().values
    avg_loss = pd.Series(loss).ewm(span=3, adjust=False, min_periods=3).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi3 = 100 - (100 / (1 + rs))
    
    # 4h RSI(2) for streak component
    delta2 = close_series.diff()
    up = np.where(delta2 > 0, delta2, 0)
    down = np.where(delta2 < 0, -delta2, 0)
    # Streak calculation: consecutive up/down days
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    # RSI of streak
    streak_series = pd.Series(streak)
    streak_delta = streak_series.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0)
    avg_streak_gain = pd.Series(streak_gain).ewm(span=2, adjust=False, min_periods=2).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=2, adjust=False, min_periods=2).mean().values
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    streak_rsi = 100 - (100 / (1 + streak_rs))
    
    # 4h Percent Rank(100) for CRSI
    def percent_rank(arr, window):
        pr = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            window_data = arr[i-window+1:i+1]
            if np.all(np.isnan(window_data)):
                pr[i] = np.nan
            else:
                pr[i] = np.sum(~np.isnan(window_data) & (window_data <= arr[i])) / np.sum(~np.isnan(window_data)) * 100
        return pr
    percent_rank_100 = percent_rank(close, 100)
    
    # CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    crsi = (rsi3 + streak_rsi + percent_rank_100) / 3.0
    
    # 1d data for Choppiness Index regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Choppiness Index on 1d
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    atr1 = d_high - d_low
    atr2 = np.abs(d_high - np.roll(d_close, 1))
    atr3 = np.abs(d_low - np.roll(d_close, 1))
    atr2[0] = atr1[0]
    atr3[0] = atr1[0]
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(d_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(d_low).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_1d = np.where((highest_high - lowest_low) == 0, 50, chop_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 4h Donchian(20) for exit
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(crsi[i]) or np.isnan(ema200[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(donchian_mid[i]) or 
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
            # Exit: price reaches Donchian midpoint or trend reverses
            elif close[i] < donchian_mid[i] or close[i] < ema200[i]:
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
            # Exit: price reaches Donchian midpoint or trend reverses
            elif close[i] > donchian_mid[i] or close[i] > ema200[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with CRSI extremes, trend filter, and range regime
            # Long: CRSI < 15, price > EMA200, Choppiness > 61.8 (range)
            if (crsi[i] < 15 and
                close[i] > ema200[i] and
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: CRSI > 85, price < EMA200, Choppiness > 61.8 (range)
            elif (crsi[i] > 85 and
                  close[i] < ema200[i] and
                  chop_1d_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA (Kaufman Adaptive Moving Average) direction with RSI filter and Choppiness regime
# Long when KAMA > KAMA_prev, RSI(14) > 50, Choppiness > 61.8 (range)
# Short when KAMA < KAMA_prev, RSI(14) < 50, Choppiness > 61.8 (range)
# Exit when price crosses KAMA in opposite direction or trend reverses
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 60-120 total trades over 4 years (15-30/year)

name = "4h_kama_rsi_chop_v1"
timeframe = "4h"
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
    
    # 4h KAMA (Kaufman Adaptive Moving Average)
    def kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        kama_out = np.full_like(close, np.nan)
        kama_out[period] = close[period]
        for i in range(period+1, len(close)):
            if np.isnan(kama_out[i-1]):
                kama_out[i] = close[i]
            else:
                kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_val = kama(close, 10, 2, 30)
    
    # 1d data for Choppiness Index regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Choppiness Index on 1d
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    atr1 = d_high - d_low
    atr2 = np.abs(d_high - np.roll(d_close, 1))
    atr3 = np.abs(d_low - np.roll(d_close, 1))
    atr2[0] = atr1[0]
    atr3[0] = atr1[0]
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(d_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(d_low).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_1d = np.where((highest_high - lowest_low) == 0, 50, chop_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # RSI(14) for filter
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(kama_val[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(atr[i])):
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
            # Exit: price crosses KAMA downward or trend reverses (RSI < 50)
            elif close[i] < kama_val[i] or rsi[i] < 50:
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
            # Exit: price crosses KAMA upward or trend reverses (RSI > 50)
            elif close[i] > kama_val[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with KAMA direction, RSI filter, and range regime
            # Long: KAMA rising, RSI > 50, Choppiness > 61.8 (range)
            if i > 0 and not np.isnan(kama_val[i-1]):
                kama_rising = kama_val[i] > kama_val[i-1]
                # Short: KAMA falling, RSI < 50, Choppiness > 61.8 (range)
                kama_falling = kama_val[i] < kama_val[i-1]
                
                if (kama_rising and
                    rsi[i] > 50 and
                    chop_1d_aligned[i] > 61.8):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif (kama_falling and
                      rsi[i] < 50 and
                      chop_1d_aligned[i] > 61.8):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

--- The strategy code must be a single code block. Make sure to output only the code. ---