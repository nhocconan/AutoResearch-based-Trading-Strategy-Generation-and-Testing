#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly trend filter and volume confirmation
# Long: price breaks above Donchian(20) high AND weekly EMA(20) trend up AND volume > 1.5x avg
# Short: price breaks below Donchian(20) low AND weekly EMA(20) trend down AND volume > 1.5x avg
# Exit: opposite Donchian break or trailing stop at 2x ATR
# Target: 50-150 trades over 4 years with 0.25 position size

name = "6h_donchian_weekly_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donch_high = high_roll.values
    donch_low = low_roll.values
    
    # Weekly EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # ATR for trailing stop (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = high[0] - low[0]
    tr3.iloc[0] = high[0] - low[0]
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_value = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: reverse signal OR trailing stop hit
            if close[i] < donch_low[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: reverse signal OR trailing stop hit
            if close[i] > donch_high[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donch_high[i] and ema_20_aligned[i] > ema_20_aligned[i-1]:
                    # Breakout above Donchian high with rising weekly trend
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    atr_value = atr[i]
                elif close[i] < donch_low[i] and ema_20_aligned[i] < ema_20_aligned[i-1]:
                    # Breakdown below Donchian low with falling weekly trend
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    atr_value = atr[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d with ADX trend filter and volume confirmation
# Fade at R3/S3 levels in ranging markets (ADX < 25), breakout continuation at R4/S4 in trending markets (ADX >= 25)
# Uses actual Camarilla formula: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
# Target: 50-150 trades over 4 years with 0.25 position size

name = "6h_camarilla_1d_adx_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    H = daily_high
    L = daily_low
    C = daily_close
    range_hl = H - L
    
    # Camarilla levels (using correct formulas)
    R4 = C + (range_hl * 1.1 / 2)
    R3 = C + (range_hl * 1.1 / 4)
    R2 = C + (range_hl * 1.1 / 6)
    R1 = C + (range_hl * 1.1 / 12)
    S1 = C - (range_hl * 1.1 / 12)
    S2 = C - (range_hl * 1.1 / 6)
    S3 = C - (range_hl * 1.1 / 4)
    S4 = C - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # ADX(14) for trend strength on 6h
    plus_dm = pd.Series(np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                                 np.maximum(high - np.roll(high, 1), 0), 0))
    minus_dm = pd.Series(np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                                  np.maximum(np.roll(low, 1) - low, 0), 0))
    plus_dm.iloc[0] = 0
    minus_dm.iloc[0] = 0
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = high[0] - low[0]
    tr3.iloc[0] = high[0] - low[0]
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    
    # Volume confirmation: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(adx_values[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price < S3 OR ADX drops below 20 (trend weakening)
            if close[i] < S3_aligned[i] or adx_values[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price > R3 OR ADX drops below 20 (trend weakening)
            if close[i] > R3_aligned[i] or adx_values[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla levels + ADX trend + volume
            if volume[i] > volume_threshold[i]:
                if adx_values[i] >= 25:  # Trending market
                    # Breakout continuation: buy at R4, sell at S4
                    if close[i] > R4_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] < S4_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging market (ADX < 25)
                    # Fade at extremes: buy at S3, sell at R3
                    if close[i] < S3_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] > R3_aligned[i]:
                        signals[i] = -0.25
                        position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku cloud system with 1d trend filter and volume confirmation
# TK cross (Tenkan/Kijun) as entry signal, cloud (Senkou A/B) as trend filter
# Long: TK cross above AND price above cloud AND volume > 1.5x avg
# Short: TK cross below AND price below cloud AND volume > 1.5x avg
# Uses proper Ichimoku calculations: Tenkan = (9-period high + low)/2, Kijun = (26-period high + low)/2
# Senkou A = (Tenkan + Kijun)/2 shifted 26 periods ahead, Senkou B = (52-period high + low)/2 shifted 26 periods ahead
# Target: 50-150 trades over 4 years with 0.25 position size

name = "6h_ichimoku_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Ichimoku calculations on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = ((high_9 + low_9) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = ((high_26 + low_26) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2).values
    
    # Daily trend filter: EMA(50) on 1d
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Wait for Senkou B to stabilize
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: TK cross below OR price below cloud OR below daily EMA
            if tenkan[i] < kijun[i] or close[i] < min(senkou_a[i], senkou_b[i]) or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TK cross above OR price above cloud OR above daily EMA
            if tenkan[i] > kijun[i] or close[i] > max(senkou_a[i], senkou_b[i]) or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross + price vs cloud + daily trend + volume
            if volume[i] > volume_threshold[i]:
                # TK cross: Tenkan crossing Kijun
                tk_cross_above = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
                tk_cross_below = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
                
                # Price vs cloud: above cloud = bullish, below cloud = bearish
                above_cloud = close[i] > max(senkou_a[i], senkou_b[i])
                below_cloud = close[i] < min(senkou_a[i], senkou_b[i])
                
                if tk_cross_above and above_cloud and close[i] > ema_50_aligned[i]:
                    # Bullish TK cross above cloud with uptrend
                    signals[i] = 0.25
                    position = 1
                elif tk_cross_below and below_cloud and close[i] < ema_50_aligned[i]:
                    # Bearish TK cross below cloud with downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with weekly trend filter and volume confirmation
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long: Bull Power > 0 AND increasing AND weekly EMA(20) up AND volume > 1.3x avg
# Short: Bear Power < 0 AND decreasing AND weekly EMA(20) down AND volume > 1.3x avg
# Exit when power crosses zero or weekly trend changes
# Target: 50-150 trades over 4 years with 0.25 position size

name = "6h_elder_ray_weekly_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA(13) for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA
    bear_power = low - ema_13   # Low - EMA
    
    # Weekly EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume confirmation: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Bull Power <= 0 OR weekly trend turns down
            if bull_power[i] <= 0 or ema_20_aligned[i] < ema_20_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bear Power >= 0 OR weekly trend turns up
            if bear_power[i] >= 0 or ema_20_aligned[i] > ema_20_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray + weekly trend + volume
            if volume[i] > volume_threshold[i]:
                # Check if powers are trending (increasing/decreasing)
                bull_increasing = bull_power[i] > bull_power[i-1]
                bear_decreasing = bear_power[i] < bear_power[i-1]
                
                if bull_power[i] > 0 and bull_increasing and ema_20_aligned[i] > ema_20_aligned[i-1]:
                    # Strong bullish momentum with rising weekly trend
                    signals[i] = 0.25
                    position = 1
                elif bear_power[i] < 0 and bear_decreasing and ema_20_aligned[i] < ema_20_aligned[i-1]:
                    # Strong bearish momentum with falling weekly trend
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Alligator lines: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
# Long: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA(50) AND volume > 1.5x avg
# Short: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA(50) AND volume > 1.5x avg
# Uses Smoothed Moving Average (SMMA) for proper Alligator calculation
# Target: 50-150 trades over 4 years with 0.25 position size

name = "6h_alligator_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 21:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Smoothed Moving Average (SMMA) function
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current_value) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines (Williams Alligator)
    jaw = smma(close, 13)  # Jaw: 13-period SMMA
    teeth = smma(close, 8)  # Teeth: 8-period SMMA
    lips = smma(close, 5)   # Lips: 5-period SMMA
    
    # Shift the lines as per Alligator specification
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that would look ahead
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Daily trend filter: EMA(50) on 1d
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Wait for Jaw to stabilize
        # Skip if required data not available
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(jaw_shifted[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Alligator alignment breaks (not Lips > Teeth > Jaw) OR price < daily EMA
            if not (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]) or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Alligator alignment breaks (not Lips < Teeth < Jaw) OR price > daily EMA
            if not (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]) or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Alligator alignment + daily trend + volume
            if volume[i] > volume_threshold[i]:
                # Bullish alignment: Lips > Teeth > Jaw
                if lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish alignment: Lips < Teeth < Jaw
                elif lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with weekly trend filter and volume confirmation
# Squeeze: BB width < 50th percentile of last 50 periods
# Breakout: price breaks above upper band OR below lower band with volume > 2x avg
# Direction: weekly EMA(20) up for long breakouts, down for short breakouts
# Target: 50-150 trades over 4 years with 0.25 position size

name = "6h_bb_squeeze_breakout_weekly_vol_v1"
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
    
    # Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    
    basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean()
    dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std()
    upper = basis + dev
    lower = basis - dev
    
    # BB Width: (Upper - Lower) / Basis
    bb_width = ((upper - lower) / basis).replace([np.inf, -np.inf], np.nan).values
    
    # Weekly EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume confirmation: volume > 2.0x 20-period average (higher threshold for breakouts)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(bb_length, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(bb_width[i]) or np.isnan(ema_20_aligned[i]) or
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to middle band OR opposite breakout
            if close[i] < basis[i] or close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to middle band OR opposite breakout
            if close[i] > basis[i] or close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: BB squeeze + breakout + weekly trend + volume
            if i >= 50:  # Need enough history for percentile calculation
                # Calculate 50th percentile of BB width over last 50 periods
                width_window = bb_width[max(0, i-50):i]
                width_50th = np.nanpercentile(width_window, 50) if not np.all(np.isnan(width_window)) else 1.0
                
                is_squeeze = bb_width[i] < width_50th
                is_breakout_up = close[i] > upper[i]
                is_breakout_down = close[i] < lower[i]
                
                if volume[i] > volume_threshold[i]:
                    if is_squeeze and is_breakout_up and ema_20_aligned[i] > ema_20_aligned[i-1]:
                        # Bullish breakout from squeeze with rising weekly trend
                        signals[i] = 0.25
                        position = 1
                    elif is_squeeze and is_breakout_down and ema_20_aligned[i] < ema_20_aligned[i-1]:
                        # Bearish breakout from squeeze with falling weekly trend
                        signals[i] = -0.25
                        position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h KAMA (Kaufman Adaptive Moving Average) with 1d regime filter and volume confirmation
# Long: price > KAMA AND market is trending (ADX > 25) AND volume > 1.3x avg
# Short: price < KAMA AND market is trending (ADX > 25) AND volume > 1.3x avg
# In ranging markets (ADX < 20): fade at 2x ATR from KAMA