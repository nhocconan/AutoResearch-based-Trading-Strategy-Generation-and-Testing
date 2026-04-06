#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# Long when price breaks above Donchian upper (20-period) AND price > 1d EMA(50) AND volume > 2x 20-period average
# Short when price breaks below Donchian lower (20-period) AND price < 1d EMA(50) AND volume > 2x 20-period average
# Exit when price crosses Donchian midline (10-period average of upper/lower)
# Uses 4h timeframe for optimal trade frequency (target 75-200/4 years), 1d EMA for trend filter, Donchian for breakout signals

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # 1-day EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate 50-period EMA on daily close
    daily_close_series = pd.Series(daily_close)
    daily_ema = daily_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align daily EMA to 4h timeframe
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(daily_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses Donchian midline
        if position == 1:  # long position
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: price breaks above Donchian upper AND price > daily EMA AND volume confirmation
            if (close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and 
                close[i] > daily_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < daily EMA AND volume confirmation
            elif (close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and 
                  close[i] < daily_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when price breaks above Donchian upper (20-period) AND price > 1w EMA(50) AND volume > 2x 20-period average
# Short when price breaks below Donchian lower (20-period) AND price < 1w EMA(50) AND volume > 2x 20-period average
# Exit when price crosses Donchian midline (10-period average of upper/lower)
# Uses 4h timeframe for optimal trade frequency (target 75-200/4 years), 1w EMA for trend filter, Donchian for breakout signals
# This strategy targets breakouts in trending markets while avoiding false breakouts in ranging markets via trend filter

name = "4h_donchian20_1w_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # 1-week EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    
    # Calculate 50-period EMA on weekly close
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema = weekly_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly EMA to 4h timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses Donchian midline
        if position == 1:  # long position
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: price breaks above Donchian upper AND price > weekly EMA AND volume confirmation
            if (close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and 
                close[i] > weekly_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < weekly EMA AND volume confirmation
            elif (close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and 
                  close[i] < weekly_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d + volume spike + choppiness regime
# Long when price touches Camarilla S3 support AND volume > 2x average AND chop > 61.8 (ranging market)
# Short when price touches Camarilla R3 resistance AND volume > 2x average AND chop > 61.8 (ranging market)
# Exit when price moves to opposite Camarilla level (S3 to R3 or R3 to S3) or chop < 38.2 (trending)
# Uses Camarilla pivot for mean reversion in ranging markets, volume for confirmation, chop for regime filter
# Target: 75-200 trades over 4 years (19-50/year) for optimal 4h performance

name = "4h_camarilla1d_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # Camarilla formulas:
    # H4 = close + 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # H2 = close + 1.1 * (high - low) / 6
    # H1 = close + 1.1 * (high - low) / 12
    # L1 = close - 1.1 * (high - low) / 12
    # L2 = close - 1.1 * (high - low) / 6
    # L3 = close - 1.1 * (high - low) / 4
    # L4 = close - 1.1 * (high - low) / 2
    # We use H3 as resistance (R3) and L3 as support (S3)
    
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close = np.roll(daily_close, 1)
    prev_daily_high[0] = np.nan
    prev_daily_low[0] = np.nan
    prev_daily_close[0] = np.nan
    
    camarilla_h3 = prev_daily_close + 1.1 * (prev_daily_high - prev_daily_low) / 4
    camarilla_l3 = prev_daily_close - 1.1 * (prev_daily_high - prev_daily_low) / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Choppiness Index (14-period)
    # Chop = 100 * log10(sum(atr(14)) / (log10(highest_high(14) - lowest_low(14)))) / log10(14)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = highest_high - lowest_low
    
    # Avoid division by zero
    chop = np.full_like(close, 50.0, dtype=float)
    mask = (range_hl > 0) & (~np.isnan(range_hl)) & (~np.isnan(atr))
    chop[mask] = 100 * np.log10(atr[mask] * 14 / range_hl[mask]) / np.log10(14)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit when price reaches resistance (H3) or chop < 38.2 (trending market)
            if close[i] >= camarilla_h3_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit when price reaches support (L3) or chop < 38.2 (trending market)
            if close[i] <= camarilla_l3_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in ranging market (chop > 61.8)
            # Long: price touches support (L3) AND volume confirmation
            if (chop[i] > 61.8 and 
                low[i] <= camarilla_l3_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches resistance (H3) AND volume confirmation
            elif (chop[i] > 61.8 and 
                  high[i] >= camarilla_h3_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h CRSI(3,2,100) with trend filter and volume confirmation
# Long when CRSI < 15 AND price > EMA(50) AND volume > 1.5x average
# Short when CRSI > 85 AND price < EMA(50) AND volume > 1.5x average
# Exit when CRSI crosses 50 (mean reversion) or opposite extreme
# Uses CRSI for mean reversion signals, EMA for trend filter, volume for confirmation
# Target: 75-200 trades over 4 years (19-50/year) for optimal 4h performance

name = "4h_crsi_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA(50) trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # CRSI calculation: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(span=3, min_periods=3, adjust=False).mean().values
    loss_ma = pd.Series(loss).ewm(span=3, min_periods=3, adjust=False).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi3 = 100 - (100 / (1 + rs))
    
    # RSI Streak(2): consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    # RSI of streak (using 2-period RSI on streak values)
    streak_change = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_change > 0, streak_change, 0)
    streak_loss = np.where(streak_change < 0, -streak_change, 0)
    streak_gain_ma = pd.Series(streak_gain).ewm(span=2, min_periods=2, adjust=False).mean().values
    streak_loss_ma = pd.Series(streak_loss).ewm(span=2, min_periods=2, adjust=False).mean().values
    streak_rs = streak_gain_ma / (streak_loss_ma + 1e-10)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # Percent Rank(100): where close ranks in last 100 periods
    pctrank = np.full(n, 50.0)
    for i in range(99, n):
        window = close[i-99:i+1]
        pctrank[i] = (np.sum(window < close[i]) + 0.5 * np.sum(window == close[i])) / len(window) * 100
    
    # CRSI
    crsi = (rsi3 + rsi_streak + pctrank) / 3
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if required data not available
        if np.isnan(ema50[i]) or np.isnan(crsi[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: CRSI crosses 50 or opposite extreme
        if position == 1:  # long position
            if crsi[i] >= 50 or crsi[i] > 85:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if crsi[i] <= 50 or crsi[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: CRSI < 15 AND price > EMA(50) AND volume confirmation
            if (crsi[i] < 15 and close[i] > ema50[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: CRSI > 85 AND price < EMA(50) AND volume confirmation
            elif (crsi[i] > 85 and close[i] < ema50[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA(10,2,30) with RSI filter and volume confirmation
# Long when KAMA direction turns up AND RSI(14) > 50 AND volume > 1.5x average
# Short when KAMA direction turns down AND RSI(14) < 50 AND volume > 1.5x average
# Exit when KAMA direction reverses or RSI crosses 50
# Uses KAMA for adaptive trend, RSI for momentum filter, volume for confirmation
# Target: 75-200 trades over 4 years (19-50/year) for optimal 4h performance

name = "4h_kama_rsi_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA calculation
    # ER = |Close - Close(10)| / Sum|Close - Close(1)| over 10 periods
    # SC = [ER * (Fastest SC - Slowest SC) + Slowest SC]^2
    # KAMA = KAMA(1) + SC * (Close - KAMA(1))
    # Fastest SC = 2/(2+1) = 0.6667, Slowest SC = 2/(30+1) = 0.0645
    
    change = np.abs(np.subtract(close[10:], close[:-10]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])).reshape(-1, 1), axis=0)
    # Proper volatility calculation for 10-period
    volatility = np.zeros(n)
    for i in range(10, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    
    # Avoid division by zero
    er = np.zeros(n)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling
    kama_dir = np.zeros(n)
    kama_dir[1:] = np.where(np.diff(kama) > 0, 1, -1)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    loss_ma = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: KAMA direction reverses or RSI crosses 50
        if position == 1:  # long position
            if kama_dir[i] == -1 or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if kama_dir[i] == 1 or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: KAMA direction up AND RSI > 50 AND volume confirmation
            if (kama_dir[i] == 1 and rsi[i] > 50 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA direction down AND RSI < 50 AND volume confirmation
            elif (kama_dir[i] == -1 and rsi[i] < 50 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Supertrend(ATR=10, mult=3) with EMA(20) filter and volume confirmation
# Long when Supertrend turns bullish AND price > EMA(20) AND volume > 2x average
# Short when Supertrend turns bearish AND price < EMA(20) AND volume > 2x average
# Exit when Supertrend reverses or price crosses EMA(20)
# Uses Supertrend for trend following, EMA for filter, volume for confirmation
# Target: 75-200 trades over 4 years (19-50/year) for optimal 4h performance

name = "4h_supertrend_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA(20) filter
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Supertrend calculation
    # ATR(10)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (3 * atr)
    lower_band = hl2 - (3 * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, n):
        if close[i] > upper_band[i-1]:
            trend[i] = 1
        elif close[i] < lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
        
        if trend[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1]) if i > 0 else lower_band[i]
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1]) if i > 0 else upper_band[i]
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(ema20[i]) or np.isnan(supertrend[i]) or np.isnan(trend[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Supertrend reverses or price crosses EMA(20)
        if position == 1:  # long position
            if trend[i] == -1 or close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if trend[i] == 1 or close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Supertrend bullish AND price > EMA(20) AND volume confirmation
            if (trend[i] == 1 and close[i] > ema20[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Supertrend bearish AND price < EMA(20) AND volume confirmation
            elif (trend[i] == -1 and close[i] < ema20[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# Long when price breaks above Donchian upper (20-period) AND price > 1d EMA(50) AND volume > 2x 20-period average
# Short when price breaks below Donchian lower (20-period) AND price < 1d EMA(50) AND volume > 2x 20-period average
# Exit when price crosses Donchian midline (10-period average of upper/lower)
# Uses 4h timeframe for optimal trade frequency (target 75-200/4 years), 1d EMA for trend filter, Donchian for breakout signals
# This strategy targets breakouts in trending markets while avoiding false breakouts in ranging markets via trend filter

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)