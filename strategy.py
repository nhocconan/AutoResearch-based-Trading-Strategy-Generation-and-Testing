#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA(50) trend filter + volume confirmation
# Long when price breaks above Donchian upper band (20-period high) AND close > 1d EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian lower band (20-period low) AND close < 1d EMA(50) AND volume > 1.5x average
# Exit when price crosses back through Donchian midline (10-period average of high/low) OR volume drops below average
# Uses 4h timeframe for optimal trade frequency, targets 75-200 total trades over 4 years
# Donchian provides clear entry/exit levels, EMA filter ensures trend alignment, volume confirms strength

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    donchian_mid = ((highest_high + lowest_low) / 2).values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: break above upper band with trend filter and volume
            if (close[i] > donchian_upper[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with trend filter and volume
            elif (close[i] < donchian_lower[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d + volume spike + choppiness regime filter
# Long when price touches Camarilla L3 support AND volume > 2x average AND CHOP > 61.8 (ranging)
# Short when price touches Camarilla H3 resistance AND volume > 2x average AND CHOP > 61.8 (ranging)
# Exit when price moves to opposite H3/L3 level OR CHOP < 38.2 (trending) OR volume drops below average
# Uses Camarilla levels for precise reversal points in ranging markets, volume for confirmation
# Targets 75-200 total trades over 4 years with strict entry conditions

name = "4h_camarilla_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h3 = np.zeros_like(daily_close)
    camarilla_l3 = np.zeros_like(daily_close)
    
    for i in range(len(daily_close)):
        if i == 0:
            # Use previous day's data (not available, use same day)
            camarilla_h3[i] = daily_close[i] + 1.1 * (daily_high[i] - daily_low[i]) / 6
            camarilla_l3[i] = daily_close[i] - 1.1 * (daily_high[i] - daily_low[i]) / 6
        else:
            camarilla_h3[i] = daily_close[i-1] + 1.1 * (daily_high[i-1] - daily_low[i-1]) / 6
            camarilla_l3[i] = daily_close[i-1] - 1.1 * (daily_high[i-1] - daily_low[i-1]) / 6
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Choppiness Index (14-period) from 1d
    tr_list = []
    for i in range(len(daily_high)):
        if i == 0:
            tr = daily_high[i] - daily_low[i]
        else:
            tr = max(daily_high[i] - daily_low[i], abs(daily_high[i] - daily_close[i-1]), abs(daily_low[i] - daily_close[i-1]))
        tr_list.append(tr)
    
    tr = np.array(tr_list)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(daily_high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(daily_low).rolling(window=14, min_periods=14).min()
    chop_1d = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_1d = chop_1d.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if (close[i] >= camarilla_h3_aligned[i] or chop_aligned[i] < 38.2 or volume[i] < volume_threshold[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if (close[i] <= camarilla_l3_aligned[i] or chop_aligned[i] < 38.2 or volume[i] < volume_threshold[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in ranging market with volume spike
            # Long: price at L3 support with volume confirmation
            if (abs(close[i] - camarilla_l3_aligned[i]) < 0.001 * camarilla_l3_aligned[i] and 
                chop_aligned[i] > 61.8 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price at H3 resistance with volume confirmation
            elif (abs(close[i] - camarilla_h3_aligned[i]) < 0.001 * camarilla_h3_aligned[i] and 
                  chop_aligned[i] > 61.8 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX(12) + volume spike + 1d ADX trend filter
# Long when TRIX crosses above zero AND volume > 2x average AND 1d ADX > 25 (trending up)
# Short when TRIX crosses below zero AND volume > 2x average AND 1d ADX > 25 (trending down)
# Exit when TRIX crosses back through zero OR volume drops below average OR ADX < 20
# Uses TRIX for momentum, volume for confirmation, ADX for trend strength
# Targets 75-200 total trades over 4 years with balanced long/short signals

name = "4h_trix_vol_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX (12-period) - triple exponential moving average
    ema1 = pd.Series(close).ewm(span=12, min_periods=12, adjust=False).mean()
    ema2 = pd.Series(ema1).ewm(span=12, min_periods=12, adjust=False).mean()
    ema3 = pd.Series(ema2).ewm(span=12, min_periods=12, adjust=False).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.fillna(0).values
    
    # 1d ADX for trend strength
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate True Range
    tr_list = []
    for i in range(len(daily_high)):
        if i == 0:
            tr = daily_high[i] - daily_low[i]
        else:
            tr = max(daily_high[i] - daily_low[i], abs(daily_high[i] - daily_close[i-1]), abs(daily_low[i] - daily_close[i-1]))
        tr_list.append(tr)
    
    tr = np.array(tr_list)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros_like(daily_high)
    minus_dm = np.zeros_like(daily_high)
    
    for i in range(1, len(daily_high)):
        up_move = daily_high[i] - daily_high[i-1]
        down_move = daily_low[i-1] - daily_low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
            
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed values
    tr_period = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
    
    # DI values
    plus_di = 100 * plus_dm_smooth / (tr_period + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_period + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.fillna(0).values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(trix[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if (trix[i] < 0 or volume[i] < volume_threshold[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if (trix[i] > 0 or volume[i] < volume_threshold[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: TRIX bullish crossover with volume and trend strength
            if i > 0 and trix[i] > 0 and trix[i-1] <= 0 and volume[i] > volume_threshold[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: TRIX bearish crossover with volume and trend strength
            elif i > 0 and trix[i] < 0 and trix[i-1] >= 0 and volume[i] > volume_threshold[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R(14) + 1d EMA(20) trend filter + volume confirmation
# Long when Williams %R crosses above -50 (bullish) AND close > 1d EMA(20) AND volume > 1.5x average
# Short when Williams %R crosses below -50 (bearish) AND close < 1d EMA(20) AND volume > 1.5x average
# Exit when Williams %R reaches opposite extreme (-20 for long, -80 for short) OR volume drops
# Uses Williams %R for momentum oscillations, EMA for trend alignment, volume for strength
# Targets 75-200 total trades over 4 years with clear entry/exit signals

name = "4h_williamsr_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    willr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    willr = willr.fillna(0).values
    
    # 1d EMA(20) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_20 = pd.Series(daily_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(willr[i]) or np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if (willr[i] >= -20 or volume[i] < volume_threshold[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if (willr[i] <= -80 or volume[i] < volume_threshold[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R crossing -50 level with trend and volume
            if i > 0:
                # Long: Williams %R crosses above -50 (bullish momentum)
                if (willr[i] > -50 and willr[i-1] <= -50 and close[i] > ema_20_aligned[i] and 
                    volume[i] > volume_threshold[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -50 (bearish momentum)
                elif (willr[i] < -50 and willr[i-1] >= -50 and close[i] < ema_20_aligned[i] and 
                      volume[i] > volume_threshold[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA (adaptive moving average) + 1d RSI + volume confirmation
# Long when price > KAMA AND RSI(1d) > 50 AND volume > 1.5x average
# Short when price < KAMA AND RSI(1d) < 50 AND volume > 1.5x average
# Exit when price crosses back through KAMA OR volume drops below average
# Uses KAMA for adaptive trend following, RSI for momentum bias, volume for strength
# Targets 75-200 total trades over 4 years with smooth trend signals

name = "4h_kama_1d_rsi_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Adaptive Moving Average) - uses efficiency ratio
    # ER = |net change| / sum(|abs change|) over lookback period
    # Smoothest constant = (ER * (fastest SC - slowest SC) + slowest SC)^2
    # where SC = 2/(period+1)
    
    fast_sc = 2 / (2 + 1)  # 2-period EMA
    slow_sc = 2 / (30 + 1)  # 30-period EMA
    
    # Calculate change and absolute change
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    
    # Calculate efficiency ratio over 10 periods
    er = np.zeros(n)
    for i in range(n):
        if i < 10:
            er[i] = 0
        else:
            net_change = abs(close[i] - close[i-10])
            total_change = np.sum(abs_change[i-9:i+1])
            er[i] = net_change / (total_change + 1e-10)
    
    # Calculate smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d RSI for momentum bias
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate RSI
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(kama[i]) or np.isnan(rsi_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if (close[i] <= kama[i] or volume[i] < volume_threshold[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if (close[i] >= kama[i] or volume[i] < volume_threshold[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price vs KAMA with RSI bias and volume
            if (close[i] > kama[i] and rsi_aligned[i] > 50 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < kama[i] and rsi_aligned[i] < 50 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Elder Ray Index (bull/bear power) + 1d EMA(13) trend filter + volume
# Long when Bull Power > 0 AND Bear Power < 0 AND close > 1d EMA(13) AND volume > 1.5x average
# Short when Bear Power > 0 AND Bull Power < 0 AND close < 1d EMA(13) AND volume > 1.5x average
# Exit when power values converge OR volume drops
# Uses Elder Ray to measure bull/bear strength relative to EMA, EMA for trend, volume for confirmation
# Targets 75-200 total trades over 4 years with clear bull/bear signals

name = "4h_elderray_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA(13) for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray Index components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # 1d EMA(13) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema13_1d = pd.Series(daily_close).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema13_1d_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or volume[i] < volume_threshold[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if (bear_power[i] <= 0 or bull_power[i] >= 0 or volume[i] < volume_threshold[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: clear bull/bear power with trend and volume
            if (bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema13_1d_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            elif (bear_power[i] > 0 and bull_power[i] < 0 and close[i] < ema13_1d_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Vortex Indicator (VI) + 1d ADX trend filter + volume confirmation
# Long when VI+ > VI- AND VI+ rising AND 1d ADX > 20 AND volume > 1.5x average
# Short when VI- > VI+ AND VI- rising AND 1d ADX > 20 AND volume > 1.5x average
# Exit when VI crossover reverses OR volume drops OR ADX < 20
# Uses Vortex to detect trend initiation, ADX for trend strength, volume for confirmation
# Targets 75-200 total trades over 4 years with clear trend signals

name = "4h_vortex_1d_adx_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Vortex Indicator (14-period)
    # VM+ = |High - Prior Low|
    # VM- = |Low - Prior High|
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.abs(high[0] - low[0])  # First period
    vm_minus[0] = np.abs(low[0] - high[0])  # First period
    
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high[0] - low[0]  # First period
    
    # Sum over 14 periods
    vi_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum() / \
              pd.Series(tr).rolling(window=14, min_periods=14).sum()
    vi_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum() / \
               pd.Series(tr).rolling(window=14, min_periods=14).sum()
    vi_plus = vi_plus.fillna(0).values
    vi_minus = vi_minus.fillna(0).values
    
    # 1d ADX for trend strength
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate True Range
    tr_list = []
    for i in range(len(daily_high)):
        if i == 0:
            tr = daily_high[i] - daily_low[i]
        else:
            tr = max(daily_high[i] - daily_low[i], abs(daily_high[i] - daily_close[i-1]), abs(daily_low[i] - daily_close[i-1]))
        tr_list.append(tr)
    
    tr = np.array(tr_list)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros_like(daily_high)
    minus_dm =