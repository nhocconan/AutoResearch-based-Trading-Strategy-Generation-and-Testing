#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses Donchian channel from prior completed 4h for structure, 1d EMA50 for trend filter
# Volume confirmation (>2.0x 20 EMA) ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# 1d EMA50 ensures we only trade with the major trend, reducing whipsaw in ranging markets.
# Works in both bull and bear by following the higher timeframe trend.

name = "4h_Donchian20_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channel (20) from prior completed 4h bar
    # We need to look back 20 completed 4h bars, so we use rolling window on 4h data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate rolling max/min for Donchian channels
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_high = rolling_max(high_4h, 20)
    donchian_low = rolling_min(low_4h, 20)
    
    # Align Donchian levels to 4h timeframe (already aligned, just need to shift for completed bar)
    # Since we're using completed 4h bars, we shift by 1 to avoid look-ahead
    donchian_high_shifted = np.roll(donchian_high, 1)
    donchian_low_shifted = np.roll(donchian_low, 1)
    donchian_high_shifted[0] = np.nan
    donchian_low_shifted[0] = np.nan
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_shifted)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + price above 1d EMA50 + volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema_50_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + price below 1d EMA50 + volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_50_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR price crosses below 1d EMA50
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            if not np.isnan(donchian_mid) and (close[i] < donchian_mid or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR price crosses above 1d EMA50
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            if not np.isnan(donchian_mid) and (close[i] > donchian_mid or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from prior completed 1d for structure, 1d EMA34 for trend filter
# Volume spike (>2.5x 20 EMA) ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# 1d EMA34 ensures we only trade with the major trend, reducing whipsaw in ranging markets.
# Works in both bull and bear by following the higher timeframe trend.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla levels from prior completed 1d bar
    # Camarilla R3, S3, R4, S4 levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Camarilla levels
    r4 = pp + ((high_1d - low_1d) * 1.1 / 2)
    r3 = pp + ((high_1d - low_1d) * 1.1 / 4)
    s3 = pp - ((high_1d - low_1d) * 1.1 / 4)
    s4 = pp - ((high_1d - low_1d) * 1.1 / 2)
    
    # Shift by 1 to use prior completed 1d bar (avoid look-ahead)
    r3_shifted = np.roll(r3, 1)
    s3_shifted = np.roll(s3, 1)
    r4_shifted = np.roll(r4, 1)
    s4_shifted = np.roll(s4, 1)
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    r4_shifted[0] = np.nan
    s4_shifted[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_shifted)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_shifted)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + price above 1d EMA34 + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_34_aligned[i] and volume[i] > (2.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + price below 1d EMA34 + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_34_aligned[i] and volume[i] > (2.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla R4/S3 midpoint OR price crosses below 1d EMA34
            camarilla_mid = (r4_aligned[i] + s3_aligned[i]) / 2.0
            if not np.isnan(camarilla_mid) and (close[i] < camarilla_mid or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla R3/S4 midpoint OR price crosses above 1d EMA34
            camarilla_mid = (r3_aligned[i] + s4_aligned[i]) / 2.0
            if not np.isnan(camarilla_mid) and (close[i] > camarilla_mid or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h CRSI(2,14,100) with 1d EMA50 trend filter and choppiness regime filter
# Uses CRSI to identify extreme overbought/oversold conditions, 1d EMA50 for trend filter
# Choppiness filter (CHOP > 61.8) ensures we only trade in ranging markets for mean reversion
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# Works in both bull and bear by using mean reversion in ranging markets and trend following in trending markets.

name = "4h_CRSI_1dEMA50_Choppiness"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate CRSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=3, adjust=False, min_periods=3).mean().values
    avg_loss = pd.Series(loss).ewm(span=3, adjust=False, min_periods=3).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100 - (100 / (1 + rs))
    
    # RSI_Streak(2): consecutive up/down days
    streak = np.zeros_like(close)
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak (2-period RSI on streak values)
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=2, adjust=False, min_periods=2).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=2, adjust=False, min_periods=2).mean().values
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # PercentRank(100): percent rank of current close over last 100 bars
    def rolling_percent_rank(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            window_data = arr[i-window+1:i+1]
            res[i] = (np.sum(window_data < arr[i]) + 0.5 * np.sum(window_data == arr[i])) / window * 100
        return res
    
    percent_rank = rolling_percent_rank(close, 100)
    
    # CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    
    # Calculate Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    # Simplified: CHOP = 100 * log10(ATR_sum / (ATR_range * log10(n))) / log10(n)
    # We'll use a simpler version: CHOP = 100 * log10(sum(True Range) / (ATR_range * log10(n))) / log10(n)
    # Actually, standard CHOP: CHOP = 100 * log10(sum(TR) / (ATR_range * log10(n))) / log10(n)
    # But we'll use a common approximation: CHOP = 100 * log10(sum(TR) / (ATR_range * log10(n))) / log10(n)
    # Let's implement the standard formula
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(1) is just TR
    # Sum of TR over 14 periods
    def rolling_sum(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.sum(arr[i-window+1:i+1])
        return res
    
    tr_sum_14 = rolling_sum(tr, 14)
    
    # ATR(14) for denominator
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_range_14 = atr_14 * 14  # approximate range over 14 periods
    
    # Choppiness Index
    chop = np.full_like(close, np.nan)
    for i in range(14, len(close)):
        if atr_range_14[i] > 0 and tr_sum_14[i] > 0:
            chop[i] = 100 * np.log10(tr_sum_14[i] / (atr_range_14[i] * np.log10(14))) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(crsi[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: CRSI < 15 (oversold) + price above 1d EMA50 + choppy market (CHOP > 61.8)
            if crsi[i] < 15.0 and close[i] > ema_50_aligned[i] and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short conditions: CRSI > 85 (overbought) + price below 1d EMA50 + choppy market (CHOP > 61.8)
            elif crsi[i] > 85.0 and close[i] < ema_50_aligned[i] and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: CRSI > 70 (overbought) OR price crosses below 1d EMA50
            if crsi[i] > 70.0 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: CRSI < 30 (oversold) OR price crosses above 1d EMA50
            if crsi[i] < 30.0 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA(10,2,30) direction with 1d RSI(14) filter and choppiness regime filter
# Uses KAMA to identify adaptive trend direction, 1d RSI for extreme conditions filter
# Choppiness filter (CHOP > 61.8) ensures we only trade in ranging markets for mean reversion
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# Works in both bull and bear by using trend following in trending markets and mean reversion in ranging markets.

name = "4h_KAMA_1dRSI_Choppiness"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI(14) filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Calculate KAMA(10,2,30)
    # ER = abs(close - close[10]) / sum(abs(close - close[1]) for i=1 to 10)
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prev_KAMA + SC * (close - prev_KAMA)
    
    # Change over 10 periods
    change = np.abs(np.roll(close, 10) - close)
    change[0:10] = 0  # first 10 bars
    
    # Volatility over 10 periods
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # Actually, we need rolling sum of absolute changes
    abs_changes = np.abs(np.diff(close, prepend=close[0]))
    abs_changes[0] = 0
    
    def rolling_sum(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.sum(arr[i-window+1:i+1])
        return res
    
    volatility_sum = rolling_sum(abs_changes, 10)
    
    # Efficiency Ratio
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if volatility_sum[i] > 0:
            er[i] = change[i] / volatility_sum[i]
        else:
            er[i] = 0
    
    # Smoothing Constants
    fastest = 2.0 / (2.0 + 1.0)  # 0.6667
    slowest = 2.0 / (30.0 + 1.0)  # 0.0625
    sc = (er * (fastest - slowest) + slowest) ** 2
    
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate Choppiness Index (CHOP) for regime filter
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(1) is just TR
    # Sum of TR over 14 periods
    def rolling_sum(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.sum(arr[i-window+1:i+1])
        return res
    
    tr_sum_14 = rolling_sum(tr, 14)
    
    # ATR(14) for denominator
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_range_14 = atr_14 * 14  # approximate range over 14 periods
    
    # Choppiness Index
    chop = np.full_like(close, np.nan)
    for i in range(14, len(close)):
        if atr_range_14[i] > 0 and tr_sum_14[i] > 0:
            chop[i] = 100 * np.log10(tr_sum_14[i] / (atr_range_14[i] * np.log10(14))) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_14_aligned[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price above KAMA + RSI < 30 (oversold) + choppy market (CHOP > 61.8)
            if close[i] > kama[i] and rsi_14_aligned[i] < 30.0 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short conditions: price below KAMA + RSI > 70 (overbought) + choppy market (CHOP > 61.8)
            elif close[i] < kama[i] and rsi_14_aligned[i] > 70.0 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA OR RSI > 70 (overbought)
            if close[i] < kama[i] or rsi_14_aligned[i] > 70.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA OR RSI < 30 (oversold)
            if close[i] > kama[i] or rsi_14_aligned[i] < 30.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses Donchian channel from prior completed 4h for structure, 1d EMA50 for trend filter
# Volume confirmation (>2.0x 20 EMA) ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# 1d EMA50 ensures we only trade with the major trend, reducing whipsaw in ranging markets.
# Works in both bull and bear by following the higher timeframe trend.

name = "4h_Donchian20_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channel (20) from prior completed 4h bar
    # We need to look back 20 completed 4h bars, so we use rolling window on 4h data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate rolling max/min for Donchian channels
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_high = rolling_max(high_4h, 20)
    donchian_low = rolling_min(low_4h, 20)
    
    # Align Donchian levels to 4h timeframe (already aligned, just need to shift for completed bar)
    # Since we're using completed 4h bars, we shift by 1 to avoid look-ahead
    donchian_high_shifted = np.roll(donchian_high, 1)
    don