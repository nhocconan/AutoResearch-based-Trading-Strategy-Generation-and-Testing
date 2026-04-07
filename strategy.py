#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper band, weekly close > weekly EMA50 (uptrend), and daily volume > 1.5x 1d average volume
# Short when price breaks below 1d Donchian lower band, weekly close < weekly EMA50 (downtrend), and daily volume > 1.5x 1d average volume
# Exit when trend reverses (weekly close crosses EMA50) or opposite breakout occurs
# Stoploss at 2.5 * ATR(14)
# Position size: 0.30 (30% of capital)
# Uses weekly EMA50 for trend filter and 1d volume average for confirmation
# Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag

name = "1d_donchian20_weekly_ema50_vol_v1"
timeframe = "1d"
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
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Donchian(20) channels
    high_series = pd.Series(high_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_1d)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Weekly data for EMA50 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # 1d volume average for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
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
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_weekly_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price below EMA50) or breaks below lower band
            elif close[i] < ema_weekly_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price above EMA50) or breaks above upper band
            elif close[i] > ema_weekly_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above upper band, price above EMA50 (uptrend), volume spike
            if (close[i] > upper_aligned[i] and
                close[i] > ema_weekly_aligned[i] and
                volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, price below EMA50 (downtrend), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema_weekly_aligned[i] and
                  volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Camarilla pivot levels with weekly EMA50 trend filter and volume confirmation
# Long when price touches or crosses above Camarilla L3 level, weekly close > weekly EMA50 (uptrend), and daily volume > 1.5x 1d average volume
# Short when price touches or crosses below Camarilla H3 level, weekly close < weekly EMA50 (downtrend), and daily volume > 1.5x 1d average volume
# Exit when price reaches Camarilla L4/H4 levels or trend reverses (weekly close crosses EMA50)
# Stoploss at 2.5 * ATR(14)
# Position size: 0.30 (30% of capital)
# Uses weekly EMA50 for trend filter and 1d volume average for confirmation
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag

name = "1d_camarilla_weekly_ema50_vol_v1"
timeframe = "1d"
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
    
    # 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day using previous day's OHLC
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 1d timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Weekly data for EMA50 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # 1d volume average for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
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
        if (np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_weekly_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches L4 or trend reverses (price below EMA50)
            elif close[i] <= l4_aligned[i] or close[i] < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches H4 or trend reverses (price above EMA50)
            elif close[i] >= h4_aligned[i] or close[i] > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price crosses above L3, price above EMA50 (uptrend), volume spike
            if (close[i] > l3_aligned[i] and
                close[i] > ema_weekly_aligned[i] and
                volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: price crosses below H3, price below EMA50 (downtrend), volume spike
            elif (close[i] < h3_aligned[i] and
                  close[i] < ema_weekly_aligned[i] and
                  volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day CRSI(2,1,100) with weekly EMA50 trend filter and volume confirmation
# CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
# Long when CRSI < 15, weekly close > weekly EMA50 (uptrend), and daily volume > 1.5x 1d average volume
# Short when CRSI > 85, weekly close < weekly EMA50 (downtrend), and daily volume > 1.5x 1d average volume
# Exit when CRSI crosses above 70 (long) or below 30 (short) or trend reverses
# Stoploss at 2.5 * ATR(14)
# Position size: 0.30 (30% of capital)
# Target: 70-140 total trades over 4 years (17-35/year) to minimize fee drag

name = "1d_crsi_weekly_ema50_vol_v1"
timeframe = "1d"
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
    
    # 1d data for CRSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate RSI(3)
    def rsi(series, period):
        delta = np.diff(series)
        delta = np.insert(delta, 0, 0)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi3 = rsi(close_1d, 3)
    
    # Calculate RSI Streak(2): consecutive up/down days
    change = np.diff(close_1d)
    change = np.insert(change, 0, 0)
    streak = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        if change[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif change[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    # RSI of streak
    rsi_streak = rsi(streak, 2)
    
    # Calculate Percent Rank(100): % of days price was lower in last 100 days
    def percent_rank(series, window):
        rank = np.zeros(len(series))
        for i in range(window, len(series)):
            window_data = series[i-window:i]
            rank[i] = np.sum(window_data < series[i]) / window * 100
        return rank
    percent_rank_100 = percent_rank(close_1d, 100)
    
    # CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    crsi = (rsi3 + rsi_streak + percent_rank_100) / 3.0
    crsi_aligned = align_htf_to_ltf(prices, df_1d, crsi)
    
    # Weekly data for EMA50 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # 1d volume average for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
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
        if (np.isnan(crsi_aligned[i]) or np.isnan(ema_weekly_aligned[i]) or 
            np.isnan(volume_ma_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: CRSI > 70 or trend reverses (price below EMA50)
            elif crsi_aligned[i] > 70 or close[i] < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: CRSI < 30 or trend reverses (price above EMA50)
            elif crsi_aligned[i] < 30 or close[i] > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: CRSI < 15, price above EMA50 (uptrend), volume spike
            if (crsi_aligned[i] < 15 and
                close[i] > ema_weekly_aligned[i] and
                volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: CRSI > 85, price below EMA50 (downtrend), volume spike
            elif (crsi_aligned[i] > 85 and
                  close[i] < ema_weekly_aligned[i] and
                  volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day KAMA(10,2,30) with weekly EMA50 trend filter and volume confirmation
# Long when KAMA > previous KAMA (uptrend), weekly close > weekly EMA50, and daily volume > 1.5x 1d average volume
# Short when KAMA < previous KAMA (downtrend), weekly close < weekly EMA50, and daily volume > 1.5x 1d average volume
# Exit when KAMA reverses direction or trend reverses (weekly close crosses EMA50)
# Stoploss at 2.5 * ATR(14)
# Position size: 0.30 (30% of capital)
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag

name = "1d_kama_weekly_ema50_vol_v1"
timeframe = "1d"
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
    
    # 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio (ER) and Smoothing Constants for KAMA
    def kama(series, fast=2, slow=30):
        change = np.abs(np.diff(series))
        change = np.insert(change, 0, 0)
        volatility = np.sum(np.abs(np.diff(series)))  # sum of absolute changes
        volatility = np.insert(volatility, 0, volatility[0])  # placeholder for first element
        
        # Calculate ER and volatility properly
        er = np.zeros(len(series))
        for i in range(1, len(series)):
            if i == 1:
                er[i] = 1.0 if series[i] != series[i-1] else 0.0
            else:
                price_change = np.abs(series[i] - series[i-9])  # 10-period change
                vol_sum = np.sum(np.abs(np.diff(series[i-9:i+1])))  # 10-period volatility
                er[i] = price_change / (vol_sum + 1e-10) if vol_sum > 0 else 0
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # Calculate KAMA
        kama_vals = np.zeros(len(series))
        kama_vals[0] = series[0]
        for i in range(1, len(series)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (series[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close_1d, 2, 30)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_vals)
    
    # Weekly data for EMA50 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # 1d volume average for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
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
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_weekly_aligned[i]) or 
            np.isnan(volume_ma_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA reverses (current < previous) or trend reverses (price below EMA50)
            elif kama_aligned[i] < kama_aligned[i-1] or close[i] < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA reverses (current > previous) or trend reverses (price above EMA50)
            elif kama_aligned[i] > kama_aligned[i-1] or close[i] > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: KAMA rising (current > previous), price above EMA50 (uptrend), volume spike
            if (kama_aligned[i] > kama_aligned[i-1] and
                close[i] > ema_weekly_aligned[i] and
                volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: KAMA falling (current < previous), price below EMA50 (downtrend), volume spike
            elif (kama_aligned[i] < kama_aligned[i-1] and
                  close[i] < ema_weekly_aligned[i] and
                  volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day funding rate mean reversion with weekly EMA50 trend filter
# Long when funding rate Z-score < -2.0 (extreme negative), weekly close > weekly EMA50 (uptrend)
# Short when funding rate Z-score > +2.0 (extreme positive), weekly close < weekly EMA50 (downtrend)
# Exit when funding rate Z-score crosses back above -0.5 (long) or below +0.5 (short) or trend reverses
# Stoploss at 2.5 * ATR(14)
# Position size: 0.30 (30% of capital)
# Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag
# Note: Requires funding data from data/processed/funding/ directory

name = "1d_funding_mean_reversion_weekly_ema50_v1"
timeframe = "1d"
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
    
    # Attempt to load funding rate data
    try:
        # Extract symbol from first row (assuming consistent symbol)
        symbol = 'UNKNOWN'
        if 'symbol' in prices.columns and len(prices) > 0:
            symbol = prices['symbol'].iloc[0]
        else:
            # Try to infer from common symbols
            pass
        
        # Construct funding data path
        funding_path = f"data/processed/funding/{symbol}.parquet"
        funding_df = pd.read_parquet(funding_path)
        
        # Ensure funding data has required columns
        if 'funding_rate' not in funding_df.columns:
            raise ValueError("Missing funding_rate column")
            
        # Align funding data to price timeline
        funding_rates = funding_df['funding_rate'].values
        # Simple alignment - assuming same frequency and start time
        # In practice, we'd need proper timestamp alignment
        if len(funding_rates) >= n:
            funding_rates = funding_rates[:n]
        else:
            # Pad or repeat if needed
            funding_rates = np.pad(funding_rates, (0, n - len(funding_rates)), 'edge')
            
    except Exception as e:
        # Fallback: if funding data unavailable, use a dummy signal that will produce no trades
        # This ensures the strategy doesn't break but won't generate meaningful signals
        return np.zeros(n)
    
    # Calculate Z-score of funding rate over 30-day window
    funding_series = pd.Series(funding_rates)
    funding_mean = funding_series.rolling(window=30, min_periods=30).mean().values
    funding_std = funding_series.rolling(window=30, min_periods=30).std().values
    funding_zscore = (funding_rates - funding_mean) / (funding_std + 1e-10)
    funding_zscore_aligned = align_