#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d trend filter and volume confirmation
# In bull markets: buy breakouts above 20-period high when 1d EMA50 > EMA200
# In bear markets: sell breakdowns below 20-period low when 1d EMA50 < EMA200
# Volume must exceed 1.5x 20-period average to confirm breakout
# Designed to work in both regimes by following the higher timeframe trend
# Target: 50-150 trades over 4 years by requiring multiple confirmations

name = "6h_donchian_1d_trend_vol_v1"
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
    
    # Donchian channels (20-period) on 6h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # 1d trend filter: EMA50 vs EMA200
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price closes below Donchian low OR trend reverses
            if close[i] <= donchian_low[i] or ema_50_aligned[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian high OR trend reverses
            if close[i] >= donchian_high[i] or ema_50_aligned[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend alignment + volume
            if volume[i] > volume_threshold[i]:
                # Long: breakout above Donchian high in uptrend
                if close[i] > donchian_high[i] and ema_50_aligned[i] > ema_200_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: breakdown below Donchian low in downtrend
                elif close[i] < donchian_low[i] and ema_50_aligned[i] < ema_200_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d with ADX filter and volume confirmation
# Fade at R3/S3 levels when price reverts to mean in ranging markets (ADX < 25)
# Breakout continuation at R4/S4 levels when trending (ADX > 25)
# Volume must exceed 1.3x average to confirm the move
# Works in both bull/bear by adapting to market regime via ADX
# Target: 50-150 trades over 4 years with balanced long/short

name = "6h_camarilla_1d_adx_vol_v1"
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
    
    # Calculate 1d Camarilla levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    camarilla_h5 = prev_close + 1.1 * range_ / 2  # R4
    camarilla_h4 = prev_close + 1.1 * range_ / 4  # R3
    camarilla_h3 = prev_close + 1.1 * range_ / 6  # R2
    camarilla_l3 = prev_close - 1.1 * range_ / 6  # S2
    camarilla_l2 = prev_close - 1.1 * range_ / 4  # S3
    camarilla_l1 = prev_close - 1.1 * range_ / 2  # S4
    
    # Align levels to 6h timeframe
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # ADX(14) on 1d for trend strength
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d_series - low_1d_series
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Movement
    up_move = high_1d_series - high_1d_series.shift(1)
    down_move = low_1d_series.shift(1) - low_1d_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
    
    # DI and DX
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(h4_aligned[i]) or np.isnan(l2_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price reaches H3 (take profit) OR reverses below L2 (stop)
            if close[i] >= h3_aligned[i] or close[i] <= l2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches L3 (take profit) OR reverses above H2 (stop)
            if close[i] <= l3_aligned[i] or close[i] >= h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on ADX regime
            if volume[i] > volume_threshold[i]:
                # Ranging market (ADX < 25): fade at H4/L2
                if adx_aligned[i] < 25:
                    if close[i] <= h4_aligned[i] and close[i] > h4_aligned[i-1]:
                        # Rejection at H4 (R3) - go short
                        signals[i] = -0.25
                        position = -1
                    elif close[i] >= l2_aligned[i] and close[i] < l2_aligned[i-1]:
                        # Rejection at L2 (S3) - go long
                        signals[i] = 0.25
                        position = 1
                # Trending market (ADX >= 25): breakout at H5/L1
                else:
                    if close[i] > h5_aligned[i]:
                        # Breakout above H5 (R4) - go long
                        signals[i] = 0.25
                        position = 1
                    elif close[i] < l1_aligned[i]:
                        # Breakdown below L1 (S4) - go short
                        signals[i] = -0.25
                        position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku cloud system with 1d filter and volume confirmation
# Enter long when: Tenkan > Kijun AND price above cloud AND 1d trend bullish
# Enter short when: Tenkan < Kijun AND price below cloud AND 1d trend bearish
# Volume must exceed 1.4x average to confirm the signal
# Cloud acts as dynamic support/resistance, TK cross as momentum signal
# Works in both bull/bear by requiring alignment with higher timeframe trend
# Target: 60-180 trades over 4 years

name = "6h_ichimoku_1d_trend_vol_v1"
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
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = (high_52 + low_52) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals to avoid look-ahead
    
    # 1d trend filter: EMA50 vs EMA200
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: volume > 1.4x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.4 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Wait for Senkou B to be calculated
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou A and B shifted forward 26 periods)
        # For signal at time i, we use Senkou A/B from i-26 (already published)
        if i >= 26:
            senkou_a_val = senkou_a[i-26]
            senkou_b_val = senkou_b[i-26]
        else:
            # Not enough data for cloud
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        if position == 1:  # long position
            # Exit: price falls below cloud OR TK cross turns bearish
            if close[i] < cloud_bottom or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above cloud OR TK cross turns bullish
            if close[i] > cloud_top or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross + price vs cloud + 1d trend + volume
            if volume[i] > volume_threshold[i]:
                # Bullish: TK cross bullish + price above cloud + 1d uptrend
                if (tenkan[i] > kijun[i] and 
                    close[i] > cloud_top and 
                    ema_50_aligned[i] > ema_200_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Bearish: TK cross bearish + price below cloud + 1d downtrend
                elif (tenkan[i] < kijun[i] and 
                      close[i] < cloud_bottom and 
                      ema_50_aligned[i] < ema_200_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with weekly trend filter and volume confirmation
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Enter long when: Bull Power > 0 AND increasing AND weekly EMA21 > EMA50
# Enter short when: Bear Power > 0 AND increasing AND weekly EMA21 < EMA50
# Volume must exceed 1.5x average to confirm the move
# Works in both bull/bear by aligning with higher timeframe trend
# Target: 50-150 trades over 4 years

name = "6h_elder_ray_weekly_trend_vol_v1"
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
    
    # Elder Ray components (EMA13)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = EMA13 - Low
    bear_power = ema13 - low
    
    # Weekly trend filter: EMA21 vs EMA50
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema21_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Bull Power turns negative OR weekly trend turns bearish
            if bull_power[i] <= 0 or ema21_1w_aligned[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bear Power turns negative OR weekly trend turns bullish
            if bear_power[i] <= 0 or ema21_1w_aligned[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Power > 0 AND increasing + weekly trend + volume
            if volume[i] > volume_threshold[i]:
                # Long: Bull Power positive AND rising AND weekly uptrend
                if (bull_power[i] > 0 and 
                    bull_power[i] > bull_power[i-1] and 
                    ema21_1w_aligned[i] > ema50_1w_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power positive AND rising AND weekly downtrend
                elif (bear_power[i] > 0 and 
                      bear_power[i] > bear_power[i-1] and 
                      ema21_1w_aligned[i] < ema50_1w_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator combination with volume confirmation
# Alligator: Jaw (13), Teeth (8), Lips (5) SMAs shifted
# Enter long when: Price > Alligator lines AND ADX > 25 AND DI+ > DI-
# Enter short when: Price < Alligator lines AND ADX > 25 AND DI- > DI+
# Volume must exceed 1.3x average to confirm
# Alligator identifies trend, ADX filters strength, DI gives direction
# Works in both bull/bear by requiring strong trending conditions
# Target: 40-120 trades over 4 years

name = "6h_adx_alligator_vol_v1"
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
    
    # Williams Alligator (SMAs with shift)
    # Jaw: 13-period SMA shifted 8 bars
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)  # shift 8 bars future
    
    # Teeth: 8-period SMA shifted 5 bars
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)  # shift 5 bars future
    
    # Lips: 5-period SMA shifted 3 bars
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)  # shift 3 bars future
    
    # ADX(14) for trend strength
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Movement
    up_move = high_series - high_series.shift(1)
    down_move = low_series.shift(1) - low_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
    
    # DI and DX
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    
    # Volume confirmation: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Need at least 13 for Jaw calculation
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx_values[i]) or np.isnan(plus_di.values[i]) if hasattr(plus_di, 'values') else np.isnan(plus_di[i]) or 
            np.isnan(minus_di.values[i]) if hasattr(minus_di, 'values') else np.isnan(minus_di[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Get DI values safely
        plus_di_val = plus_di.iloc[i] if hasattr(plus_di, 'iloc') else plus_di[i]
        minus_di_val = minus_di.iloc[i] if hasattr(minus_di, 'iloc') else minus_di[i]
        
        if position == 1:  # long position
            # Exit: price falls below Alligator lines OR ADX weakens
            if close[i] < lips[i] or adx_values[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above Alligator lines OR ADX weakens
            if close[i] > lips[i] or adx_values[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Price vs Alligator + ADX strong + DI direction + volume
            if volume[i] > volume_threshold[i]:
                # Long: Price above Alligator AND ADX > 25 AND DI+ > DI-
                if (close[i] > lips[i] and close[i] > teeth[i] and close[i] > jaw[i] and
                    adx_values[i] > 25 and plus_di_val > minus_di_val):
                    signals[i] = 0.25
                    position = 1
                # Short: Price below Alligator AND ADX > 25 AND DI- > DI+
                elif (close[i] < lips[i] and close[i] < teeth[i] and close[i] < jaw[i] and
                      adx_values[i] > 25 and minus_di_val > plus_di_val):
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Connors RSI (CRSI) with 1d trend filter and volume confirmation
# CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
# Enter long when: CRSI < 15 AND price > SMA200 AND 1d EMA50 > EMA200
# Enter short when: CRSI > 85 AND price < SMA200 AND 1d EMA50 < EMA200
# Volume must exceed 1.4x average to confirm
# CRSI captures extreme mean reversion, SMA200 filters direction
# Works in both bull/bear by requiring alignment with higher timeframe trend
# Target: 60-180 trades over 4 years

name = "6h_crsi_1d_trend_vol_v1"
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
    
    # RSI(3) for CRSI
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/3, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/3, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_3 = 100 - (100 / (1 + rs))
    rsi_3 = rsi_3.values
    
    # RSI Streak(2): 2-period RSI of up/down days
    up_days = (close_series > close_series.shift(1)).astype(int)
    down_days = (close_series < close_series.shift(1)).astype(int)
    
    # RSI of up days streak
    up_streak = up_days.rolling(window=2, min_periods=2).sum()
    down_streak = down_days.rolling(window=2, min_periods=2).sum()
    
    # Calculate RSI for streak values (0,1,2)
    streak_values = up_streak - down_streak  # -2 to +2 range
    # Normalize to 0-100 for RSI calculation
    streak_normalized = (streak_values + 2) * 25  # -2->0, -1->25, 0->50, 1->75, 2->100
    
    # RSI of streak
    delta_streak = pd.Series(streak_normalized).diff()
    gain_streak = delta_streak.clip(lower=0)
    loss_streak = -delta_streak.clip(upper=0)
    avg_gain_streak = gain_streak.ewm(alpha=1/2, adjust=False).mean()
    avg_loss_streak = loss_streak.ewm(alpha=1/2, adjust=False).mean()
    rs_streak = avg_gain_streak / avg_loss_streak
    rsi_streak = 100 - (100 / (1 +