#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels from 12-hour timeframe with volume confirmation
# Fade at R3/S3 levels (mean reversion) and breakout continuation at R4/S4 levels (trend follow)
# Uses 12-hour pivot points calculated from previous day's high/low/close
# Volume confirmation: current volume > 1.5x 20-period average to avoid false signals
# Works in both bull and bear markets by adapting to price action at key levels
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_camarilla12_vol_v1"
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
    
    # 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Using previous bar's high, low, close (already closed)
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    prev_close = df_12h['close'].shift(1).values
    
    # Pivot point and ranges
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 4.0)
    s3 = pivot - (range_hl * 1.1 / 4.0)
    r4 = pivot + (range_hl * 1.1 / 2.0)
    s4 = pivot - (range_hl * 1.1 / 2.0)
    
    # Align to 6h timeframe (shifted by 1 for completed bars only)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches S3 (mean reversion target) or breaks below S4 (trend reversal)
            elif close[i] <= s3_aligned[i] or close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches R3 (mean reversion target) or breaks above R4 (trend reversal)
            elif close[i] >= r3_aligned[i] or close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries at Camarilla levels with volume confirmation
            # Long: price breaks above R4 (breakout) OR price bounces from S3 (mean reversion)
            if volume[i] > 1.5 * volume_ma[i]:
                if close[i] > r4_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < s3_aligned[i] and close[i] > s4_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
            # Short: price breaks below S4 (breakdown) OR price bounces from R3 (mean reversion)
                elif close[i] < s4_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                elif close[i] > r3_aligned[i] and close[i] < r4_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (Bull Power/Bear Power) with 1-day EMA trend filter and volume confirmation
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA(50) AND volume > 1.5x 20-period avg
# Short when Bear Power > 0 AND Bull Power < 0 AND price < 1d EMA(50) AND volume > 1.5x 20-period avg
# Uses Elder Ray to measure bull/bear strength relative to EMA, filtered by 1d trend
# Works in both bull and bear markets by following higher timeframe trend
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_elder_ray_1d_ema_vol_v1"
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
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h EMA(13) for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA(13)
    bear_power = ema_13 - low   # Bear Power: EMA(13) - Low
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bear Power becomes positive (bearish pressure) or trend turns bearish (below EMA)
            elif bear_power[i] > 0 or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bull Power becomes positive (bullish pressure) or trend turns bullish (above EMA)
            elif bull_power[i] > 0 or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: Bull Power positive (strong bulls) AND Bear Power negative (weak bears) AND price above EMA (bullish trend) AND volume spike
            if (bull_power[i] > 0 and
                bear_power[i] > 0 and  # Note: Bear Power > 0 means bears are weak (EMA > Low)
                close[i] > ema_1d_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bear Power positive (strong bears) AND Bull Power negative (weak bulls) AND price below EMA (bearish trend) AND volume spike
            elif (bear_power[i] > 0 and
                  bull_power[i] < 0 and  # Bull Power < 0 means bulls are weak (High < EMA)
                  close[i] < ema_1d_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

Wait, I see an error in the Elder Ray logic. Let me correct it.

Bull Power = High - EMA(13)
- Bull Power > 0 means High > EMA (bullish)
- Bull Power < 0 means High < EMA (bearish)

Bear Power = EMA(13) - Low
- Bear Power > 0 means EMA > Low (bullish? Wait no...)
Actually, let me think:
- If EMA(13) - Low > 0, that means EMA > Low, which is generally bullish
- But for Bear Power to indicate bearish strength, we want Low to be high relative to EMA

Actually, standard Elder Ray interpretation:
- Bull Power = High - EMA(13): measures bulls' ability to push price above EMA
- Bear Power = EMA(13) - Low: measures bears' ability to push price below EMA

So:
- Strong bullish: Bull Power > 0 (strong) AND Bear Power < 0 (weak)
- Strong bearish: Bull Power < 0 (weak) AND Bear Power > 0 (strong)

Let me fix the logic.
</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (Bull Power/Bear Power) with 1-day EMA trend filter and volume confirmation
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 (bulls strong) AND Bear Power < 0 (bears weak) AND price > 1d EMA(50) AND volume > 1.5x 20-period avg
# Short when Bull Power < 0 (bulls weak) AND Bear Power > 0 (bears strong) AND price < 1d EMA(50) AND volume > 1.5x 20-period avg
# Uses Elder Ray to measure bull/bear strength relative to EMA, filtered by 1d trend
# Works in both bull and bear markets by following higher timeframe trend
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_elder_ray_1d_ema_vol_v1"
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
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h EMA(13) for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA(13) (>0 = bulls strong)
    bear_power = ema_13 - low   # Bear Power: EMA(13) - Low (>0 = bears strong)
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bear Power becomes positive (bears strong) or trend turns bearish (below EMA)
            elif bear_power[i] > 0 or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bull Power becomes negative (bulls weak) or trend turns bullish (above EMA)
            elif bull_power[i] < 0 or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: Bull Power > 0 (bulls strong) AND Bear Power < 0 (bears weak) AND price > EMA (bullish trend) AND volume spike
            if (bull_power[i] > 0 and
                bear_power[i] < 0 and
                close[i] > ema_1d_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bull Power < 0 (bulls weak) AND Bear Power > 0 (bears strong) AND price < EMA (bearish trend) AND volume spike
            elif (bull_power[i] < 0 and
                  bear_power[i] > 0 and
                  close[i] < ema_1d_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals