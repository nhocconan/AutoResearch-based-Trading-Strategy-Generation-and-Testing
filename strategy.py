#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume spike and 1w EMA trend filter
# - Long when Williams %R(14) < -80 (oversold) AND 1d volume > 1.5x 20-period average AND 1w close > 1w EMA20 (uptrend)
# - Short when Williams %R(14) > -20 (overbought) AND 1d volume > 1.5x 20-period average AND 1w close < 1w EMA20 (downtrend)
# - Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Williams %R identifies exhaustion points in ranging markets
# - Volume spike confirms participation
# - Weekly EMA filter ensures alignment with higher timeframe trend
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_1w_williamsr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full_like(close, np.nan, dtype=float)
    for i in range(13, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Pre-compute 1w EMA20
    close_1w = df_1w['close'].values
    ema_20_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])  # SMA seed
        multiplier = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * multiplier) + (ema_20_1w[i-1] * (1 - multiplier))
    
    # Align HTF indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, prices, williams_r)  # same timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_series = prices['volume'].values
        vol_ma_4h = np.full_like(vol_series, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_4h[j] = np.mean(vol_series[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_4h[i]) and vol_series[i] > 1.5 * vol_ma_4h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: oversold AND volume spike AND 1w uptrend
            if (williams_r_aligned[i] < -80 and vol_spike and close[i] > ema_20_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: overbought AND volume spike AND 1w downtrend
            elif (williams_r_aligned[i] > -20 and vol_spike and close[i] < ema_20_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses -50 (mean reversion completion)
            exit_long = (position == 1 and williams_r_aligned[i] > -50)
            exit_short = (position == -1 and williams_r_aligned[i] < -50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter
# - Long when price breaks above H3 pivot AND 1d volume > 1.3x 20-period average AND 1w close > 1w EMA50 (uptrend)
# - Short when price breaks below L3 pivot AND 1d volume > 1.3x 20-period average AND 1w close < 1w EMA50 (downtrend)
# - Exit when price returns to P pivot level (mean reversion)
# - Uses discrete position sizing 0.30 for optimal risk/return
# - Camarilla levels provide high-probability reversal points
# - Volume spike confirms breakout validity
# - Weekly EMA filter ensures alignment with higher timeframe trend
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_1w_camarilla_pivot_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    camarilla_h3 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_l3 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_p = np.full_like(close_1d, np.nan, dtype=float)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1])):
            range_1d = high_1d[i-1] - low_1d[i-1]
            camarilla_p[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3
            camarilla_h3[i] = camarilla_p[i] + range_1d * 1.1 / 4
            camarilla_l3[i] = camarilla_p[i] - range_1d * 1.1 / 4
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Pre-compute 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])  # SMA seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * multiplier) + (ema_50_1w[i-1] * (1 - multiplier))
    
    # Align HTF indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_p_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Volume spike condition (1.3x average)
        vol_series = prices['volume'].values
        vol_ma_4h = np.full_like(vol_series, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_4h[j] = np.mean(vol_series[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_4h[i]) and vol_series[i] > 1.3 * vol_ma_4h[i]
        
        close_price = prices['close'].values[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: break above H3 AND volume spike AND 1w uptrend
            if (close_price > camarilla_h3_aligned[i] and vol_spike and 
                close_price > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.30
            # Short conditions: break below L3 AND volume spike AND 1w downtrend
            elif (close_price < camarilla_l3_aligned[i] and vol_spike and 
                  close_price < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to P pivot level (mean reversion)
            exit_long = (position == 1 and close_price <= camarilla_p_aligned[i])
            exit_short = (position == -1 and close_price >= camarilla_p_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.30
                else:
                    signals[i] = -0.30
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with volume confirmation and chop filter
# - Long when price breaks above Donchian(20) upper band AND volume > 1.4x 20-period average AND CHOP(14) < 45 (trending)
# - Short when price breaks below Donchian(20) lower band AND volume > 1.4x 20-period average AND CHOP(14) < 45 (trending)
# - Exit when price returns to Donchian midpoint (mean reversion)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian breakouts capture strong momentum moves
# - Volume confirmation validates breakout strength
# - Chop filter ensures we only trade in trending markets (avoid whipsaws in ranges)
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 4h volume average (20-period)
    volume = prices['volume'].values
    vol_ma = np.full_like(volume, np.nan, dtype=float)
    for i in range(19, len(volume)):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # Pre-compute 4h Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR) / (log10(highest_high - lowest_low)) / log10(n))
    atr_vals = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    
    # ATR(14)
    atr_vals = np.full_like(tr, np.nan, dtype=float)
    for i in range(13, n):
        atr_vals[i] = np.mean(tr[i-13:i+1])
    
    # Choppiness Index
    chop = np.full_like(close, np.nan, dtype=float)
    for i in range(13, n):
        if not np.isnan(atr_vals[i]):
            sum_atr = np.sum(atr_vals[i-13:i+1])
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if highest_high != lowest_low:
                chop[i] = 100 * np.log10(sum_atr) / np.log10(highest_high - lowest_low) / np.log10(14)
            else:
                chop[i] = 50  # neutral when range is zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.4x average)
        vol_spike = volume[i] > 1.4 * vol_ma[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: break above upper band AND volume spike AND trending (CHOP < 45)
            if (close[i] > donchian_high[i] and vol_spike and chop[i] < 45):
                position = 1
                signals[i] = 0.25
            # Short conditions: break below lower band AND volume spike AND trending (CHOP < 45)
            elif (close[i] < donchian_low[i] and vol_spike and chop[i] < 45):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to midpoint (mean reversion)
            exit_long = (position == 1 and close[i] <= donchian_mid[i])
            exit_short = (position == -1 and close[i] >= donchian_mid[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX momentum with volume spike and 1w trend filter
# - Long when TRIX(12) crosses above zero AND 1d volume > 1.3x 20-period average AND 1w close > 1w EMA34 (uptrend)
# - Short when TRIX(12) crosses below zero AND 1d volume > 1.3x 20-period average AND 1w close < 1w EMA34 (downtrend)
# - Exit when TRIX returns to zero (mean reversion)
# - Uses discrete position sizing 0.30 for optimal risk/return
# - TRIX filters out insignificant price moves and identifies momentum
# - Volume spike confirms participation
# - Weekly EMA filter ensures alignment with higher timeframe trend
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_1w_trix_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h TRIX(12)
    close = prices['close'].values
    
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago, then percent change
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        ema_vals = np.full_like(arr, np.nan, dtype=float)
        multiplier = 2 / (period + 1)
        ema_vals[period-1] = np.mean(arr[:period])  # SMA seed
        for i in range(period, len(arr)):
            ema_vals[i] = (arr[i] * multiplier) + (ema_vals[i-1] * (1 - multiplier))
        return ema_vals
    
    ema1 = ema(close, 12)
    ema2 = ema(ema1, 12)
    ema3 = ema(ema2, 12)
    
    # TRIX = (today's EMA3 - yesterday's EMA3) / yesterday's EMA3 * 100
    trix = np.full_like(close, np.nan, dtype=float)
    for i in range(1, len(close)):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Pre-compute 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])  # SMA seed
        multiplier = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (close_1w[i] * multiplier) + (ema_34_1w[i-1] * (1 - multiplier))
    
    # Align HTF indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, prices, trix)  # same timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Volume spike condition (1.3x average)
        vol_series = prices['volume'].values
        vol_ma_4h = np.full_like(vol_series, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_4h[j] = np.mean(vol_series[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_4h[i]) and vol_series[i] > 1.3 * vol_ma_4h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: TRIX crosses above zero AND volume spike AND 1w uptrend
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and vol_spike and 
                prices['close'].values[i] > ema_34_1w_aligned[i]):
                position = 1
                signals[i] = 0.30
            # Short conditions: TRIX crosses below zero AND volume spike AND 1w downtrend
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and vol_spike and 
                  prices['close'].values[i] < ema_34_1w_aligned[i]):
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: TRIX returns to zero (mean reversion)
            exit_long = (position == 1 and trix_aligned[i] <= 0)
            exit_short = (position == -1 and trix_aligned[i] >= 0)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.30
                else:
                    signals[i] = -0.30
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Elder Ray Index with volume confirmation and 1w trend filter
# - Long when Bull Power > 0 AND Bear Power < 0 AND 1d volume > 1.3x 20-period average AND 1w close > 1w EMA13 (uptrend)
# - Short when Bull Power < 0 AND Bear Power > 0 AND 1d volume > 1.3x 20-period average AND 1w close < 1w EMA13 (downtrend)
# - Exit when Elder Ray returns to zero (market balance)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Elder Ray measures bull/bear power relative to EMA
# - Volume spike confirms participation
# - Weekly EMA filter ensures alignment with higher timeframe trend
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_1w_elder_ray_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h Elder Ray Index (13-period EMA)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA(13)
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        ema_vals = np.full_like(arr, np.nan, dtype=float)
        multiplier = 2 / (period + 1)
        ema_vals[period-1] = np.mean(arr[:period])  # SMA seed
        for i in range(period, len(arr)):
            ema_vals[i] = (arr[i] * multiplier) + (ema_vals[i-1] * (1 - multiplier))
        return ema_vals
    
    ema13 = ema(close, 13)
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Pre-compute 1w EMA13
    close_1w = df_1w['close'].values
    ema_13_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 13:
        ema_13_1w[12] = np.mean(close_1w[:13])  # SMA seed
        multiplier = 2 / (13 + 1)
        for i in range(13, len(close_1w)):
            ema_13_1w[i] = (close_1w[i] * multiplier) + (ema_13_1w[i-1] * (1 - multiplier))
    
    # Align HTF indicators to 4h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, prices, bull_power)  # same timeframe
    bear_power_aligned = align_htf_to_ltf(prices, prices, bear_power)  # same timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_13_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.3x average)
        vol_series = prices['volume'].values
        vol_ma_4h = np.full_like(vol_series, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_4h[j] = np.mean(vol_series[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_4h[i]) and vol_series[i] > 1.3 * vol_ma_4h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND volume spike AND 1w uptrend
            if (bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and vol_spike and 
                prices['close'].values[i] > ema_13_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bull Power < 0 AND Bear Power > 0 AND volume spike AND 1w downtrend
            elif (bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0 and vol_spike and 
                  prices['close'].values[i] < ema_13_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Elder Ray returns to zero (market balance)
            exit_long = (position == 1 and bull_power_aligned[i] <= 0 and bear_power_aligned[i] >= 0)
            exit_short = (position == -1 and bull_power_aligned[i] >= 0 and bear_power_aligned[i] <= 0)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25