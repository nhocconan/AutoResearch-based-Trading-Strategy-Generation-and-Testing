#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels from daily timeframe with volume confirmation.
# In ranging markets (common in 2025+), price tends to revert from R3/S3 levels.
# In trending markets, breakouts through R4/S4 with volume confirmation continue the trend.
# Uses 1-day Camarilla levels for structure and volume to filter false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_camarilla1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[13] = np.mean(tr[:14])
            for i in range(14, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 1-day data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1-day bar
    # R4 = Close + (High - Low) * 1.1/2
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    # S4 = Close - (High - Low) * 1.1/2
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            range_hl = high_1d[i] - low_1d[i]
            camarilla_r4[i] = close_1d[i] + range_hl * 1.1 / 2
            camarilla_r3[i] = close_1d[i] + range_hl * 1.1 / 4
            camarilla_s3[i] = close_1d[i] - range_hl * 1.1 / 4
            camarilla_s4[i] = close_1d[i] - range_hl * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume filter: current volume > 1.3x average over last 24 periods (4 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(24, 14)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches S3 (mean reversion) or stoploss hit
            if (close[i] <= s3_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R3 (mean reversion) or stoploss hit
            if (close[i] >= r3_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price breaks above R4 with volume (breakout continuation)
            if (close[i] > r4_aligned[i] and volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below S4 with volume (breakout continuation)
            elif (close[i] < s4_aligned[i] and volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            # Long mean reversion: price touches S3 with volume (bounce)
            elif (close[i] <= s3_aligned[i] and volume_filter and 
                  i > 0 and close[i-1] > s3_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short mean reversion: price touches R3 with volume (rejection)
            elif (close[i] >= r3_aligned[i] and volume_filter and 
                  i > 0 and close[i-1] < r3_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with weekly EMA(20) trend filter and volume confirmation.
# Uses weekly trend (EMA20) to filter counter-trend trades and volume to reduce false breakouts.
# Weekly timeframe adapts to both bull and bear markets by only trading in direction of higher timeframe trend.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag while maintaining statistical significance.

name = "6h_donchian20_1w_ema20_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[13] = np.mean(tr[:14])
            for i in range(14, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # 20-period EMA on 1-week timeframe (trend filter)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 18) / 20
    
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 20-period Donchian channels on 6h
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Volume filter: current volume > 1.5x average over last 24 periods (4 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(24, 20, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below EMA or stoploss hit
            if (close[i] < ema_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above EMA or stoploss hit
            if (close[i] > ema_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price breaks above Donchian high with volume and above EMA (bullish)
            if (close[i] > donch_high[i] and volume_filter and 
                close[i] > ema_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low with volume and below EMA (bearish)
            elif (close[i] < donch_low[i] and volume_filter and 
                  close[i] < ema_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (Bull Power/Bear Power) with 1-day EMA(13) trend filter.
# Elder Ray measures bull/bear power relative to EMA: Bull Power = High - EMA, Bear Power = Low - EMA.
# In trending markets, sustained Bull/Bear power indicates strength; in ranging markets, divergences signal reversals.
# Uses daily EMA for trend alignment and volume confirmation to filter weak signals.
# Target: 60-180 total trades over 4 years (15-45/year) for optimal balance of signal quality and frequency.

name = "6h_elderray1d_ema13_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[13] = np.mean(tr[:14])
            for i in range(14, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 1-day data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 13-period EMA on 1-day timeframe (trend filter)
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 13:
        ema_1d[12] = np.mean(close_1d[:13])
        for i in range(13, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 11) / 13
    
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 13-period EMA on 6h for Elder Ray
    ema_6h = np.full(n, np.nan)
    if n >= 13:
        ema_6h[12] = np.mean(close[:13])
        for i in range(13, n):
            ema_6h[i] = (close[i] * 2 + ema_6h[i-1] * 11) / 13
    
    # Elder Ray components: Bull Power = High - EMA(6h), Bear Power = Low - EMA(6h)
    bull_power = high - ema_6h
    bear_power = low - ema_6h
    
    # Volume filter: current volume > 1.4x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 14, 13)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema_aligned[i]) or 
            np.isnan(ema_6h[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.4
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Bear Power becomes positive (selling pressure) or stoploss hit
            if (bear_power[i] > 0 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull Power becomes negative (buying pressure) or stoploss hit
            if (bull_power[i] < 0 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: Bull Power > 0 (buying pressure) with volume and above daily EMA (bullish)
            if (bull_power[i] > 0 and volume_filter and 
                close[i] > ema_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bear Power < 0 (selling pressure) with volume and below daily EMA (bearish)
            elif (bear_power[i] < 0 and volume_filter and 
                  close[i] < ema_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour ADX(14) + Williams Alligator (Jaw/Teeth/Lips) with 1-day EMA(26) trend filter.
# ADX > 25 indicates trending market; Alligator lines show direction and momentum.
# In trending markets (ADX>25), trade when Alligator is aligned (Lips > Teeth > Jaw for long, reverse for short).
# In ranging markets (ADX<25), avoid trading to prevent whipsaws.
# Uses daily EMA for higher timeframe trend alignment.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_adx_alligator1d_ema26_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[13] = np.mean(tr[:14])
            for i in range(14, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 1-day data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 26-period EMA on 1-day timeframe (trend filter)
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 26:
        ema_1d[25] = np.mean(close_1d[:26])
        for i in range(26, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 24) / 26
    
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Williams Alligator on 6h: Jaw (13), Teeth (8), Lips (5) SMAs shifted
    # Jaw: 13-period SMA shifted 8 bars
    jaw = np.full(n, np.nan)
    if n >= 13:
        for i in range(13, n):
            jaw[i] = np.mean(low[i-13:i])  # Using low for Jaw as per Williams
        # Shift 8 bars forward
        jaw_shifted = np.full(n, np.nan)
        for i in range(8, n):
            jaw_shifted[i] = jaw[i-8]
        jaw = jaw_shifted
    
    # Teeth: 8-period SMA shifted 5 bars
    teeth = np.full(n, np.nan)
    if n >= 8:
        for i in range(8, n):
            teeth[i] = np.mean(high[i-8:i])  # Using high for Teeth
        # Shift 5 bars forward
        teeth_shifted = np.full(n, np.nan)
        for i in range(5, n):
            teeth_shifted[i] = teeth[i-5]
        teeth = teeth_shifted
    
    # Lips: 5-period SMA shifted 3 bars
    lips = np.full(n, np.nan)
    if n >= 5:
        for i in range(5, n):
            lips[i] = np.mean(close[i-5:i])  # Using close for Lips
        # Shift 3 bars forward
        lips_shifted = np.full(n, np.nan)
        for i in range(3, n):
            lips_shifted[i] = lips[i-3]
        lips = lips_shifted
    
    # ADX calculation (14-period)
    # +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        elif low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # True Range
    tr = np.maximum(
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    )
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # Smoothed values
    atr_14 = np.full(n, np.nan)
    plus_dm_14 = np.full(n, np.nan)
    minus_dm_14 = np.full(n, np.nan)
    
    if n >= 14:
        # Initial values
        atr_14[13] = np.nanmean(tr[1:15])
        plus_dm_14[13] = np.nansum(plus_dm[1:15])
        minus_dm_14[13] = np.nansum(minus_dm[1:15])
        
        # Wilder smoothing
        for i in range(15, n):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
            plus_dm_14[i] = (plus_dm_14[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_14[i] = (minus_dm_14[i-1] * 13 + minus_dm[i]) / 14
    
    # DI and DX
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    
    for i in range(14, n):
        if not np.isnan(atr_14[i]) and atr_14[i] != 0:
            plus_di[i] = (plus_dm_14[i] / atr_14[i]) * 100
            minus_di[i] = (minus_dm_14[i] / atr_14[i]) * 100
            dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # ADX: smoothed DX
    adx = np.full(n, np.nan)
    if n >= 28:  # Need 14 for DX + 14 for smoothing
        adx[27] = np.nanmean(dx[14:28])
        for i in range(28, n):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(28, 13, 8, 5)  # ADX + Alligator components
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: ADX < 20 (trend weakening) or Alligator misaligned or stoploss hit
            if (adx[i] < 20 or
                not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: ADX < 20 or Alligator misaligned or stoploss hit
            if (adx[i] < 20 or
                not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: ADX > 25 (trending) + Alligator aligned (Lips > Teeth > Jaw) + above daily EMA
            if (adx[i] > 25 and
                lips[i] > teeth[i] and teeth[i] > jaw[i] and
                close[i] > ema_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: ADX > 25 + Alligator aligned (Lips < Teeth < Jaw) + below daily EMA
            elif (adx[i] > 25 and
                  lips[i] < teeth[i] and teeth[i] < jaw[i] and
                  close[i] < ema_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud (Tenkan/Kijun/Senkou) with 1-day EMA(21) trend filter.
# Ichimoku provides support/resistance (cloud), momentum (TK cross), and trend (price vs cloud).
# In trending markets, price stays above/below cloud with TK cross in direction of trend.
# In ranging markets, price interacts with cloud boundaries for mean reversion.
# Uses daily EMA for higher timeframe trend alignment to avoid counter-trend trades.
# Target: 60-160 total trades over 4 years (15-40/year) for balanced frequency and accuracy.

name = "6h_ichimoku1d_ema21_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[13] = np.mean(tr[:14])
            for i in range(14, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 1-day data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 21-period EMA on 1-day timeframe (trend filter)
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 21:
        ema_1d[20] = np.mean(close_1d[:21])
        for i in range(21, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 19) / 21
    
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan = np.full(n, np.nan)
    if n >= 9:
        for i in range(9, n):
            tenkan[i] = (np.max(high[i-9:i]) + np.min(low[i-9:i])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun = np.full(n, np.nan)
    if n >= 26:
        for i in range(26, n):
            kijun[i] = (np.max(high[i-26:i]) + np.min(low[i-26:i])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = np.full(n, np.nan)
    if n >= 26:
        for i in range(26, n):
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2
    # Shift 26 periods forward
    senkou_a_shifted = np.full(n, np.nan)
    for i in range(26, n):
        senkou_a_shifted[i] = senkou_a[i-26]
    senkou_a = senkou_a_shifted
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods
    senkou_b = np.full(n, np.nan)
    if n >= 52:
        for i in range(52, n):
            senkou_b[i] = (np.max(high[i-52:i]) + np.min(low[i-52:i])) / 2
    # Shift 26 periods forward
    senkou_b_shifted = np.full(n, np.nan)
    for i in range(26, n):
        senkou_b_shifted[i] = senkou_b[i-26]
    senkou_b = senkou_b_shifted
    
    # Chikou Span (Lagging Span): close shifted -22 periods (not used for signals)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(52, 26, 9)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema_aligned[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom =