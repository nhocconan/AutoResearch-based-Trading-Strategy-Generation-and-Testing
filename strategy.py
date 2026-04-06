#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation.
# Long when RSI < 30 + 4h close > 4h EMA50 + volume > 1.5x avg.
# Short when RSI > 70 + 4h close < 4h EMA50 + volume > 1.5x avg.
# Exit when RSI crosses 50 (mean) or price crosses 4h EMA50.
# Uses 1h timeframe targeting 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear: RSI mean reversion in ranges, EMA50 filter avoids counter-trend.

name = "1h_rsi_mean_reversion_4h_ema_vol_v1"
timeframe = "1h"
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
    
    # RSI (14-period) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA50 (4h trend filter)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after RSI/EMA warmup
        # Skip if required data not available
        if np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: RSI crosses 50 or price crosses 4h EMA50
        if position == 1:  # long position
            if rsi[i] >= 50 or close[i] <= ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if rsi[i] <= 50 or close[i] >= ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for mean reversion entries with trend and volume confirmation
            # Bullish: RSI oversold + 4h uptrend + volume spike
            if (rsi[i] < 30 and 
                close[i] > ema_4h_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Bearish: RSI overbought + 4h downtrend + volume spike
            elif (rsi[i] > 70 and 
                  close[i] < ema_4h_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h price action with 4h ADX trend filter and volume confirmation.
# Long when price > 4h EMA20 + ADX > 25 + DI+ > DI- + volume > 1.5x avg.
# Short when price < 4h EMA20 + ADX > 25 + DI- > DI+ + volume > 1.5x avg.
# Exit when ADX < 20 or DI crossover reverses.
# Uses 1h timeframe targeting 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear: ADX filters ranging markets, EMA20 defines trend.

name = "1h_adx_trend_filter_vol_v1"
timeframe = "1h"
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
    
    # EMA20 (1h)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    
    # ADX (4h trend filter)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr = WilderSmooth(tr, period)
    dm_plus_smooth = WilderSmooth(dm_plus, period)
    dm_minus_smooth = WilderSmooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = WilderSmooth(dx, period)
    
    # Align ADX, DI+, DI- to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    di_plus_aligned = align_htf_to_ltf(prices, df_4h, di_plus)
    di_minus_aligned = align_htf_to_ltf(prices, df_4h, di_minus)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if required data not available
        if np.isnan(ema_20[i]) or np.isnan(adx_aligned[i]) or np.isnan(di_plus_aligned[i]) or np.isnan(di_minus_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: ADX < 20 or DI crossover reverses
        if position == 1:  # long position
            if adx_aligned[i] < 20 or di_minus_aligned[i] > di_plus_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if adx_aligned[i] < 20 or di_plus_aligned[i] > di_minus_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for trend entries with ADX filter and volume confirmation
            # Bullish: price above EMA20 + ADX strong + DI+ > DI- + volume
            if (close[i] > ema_20[i] and 
                adx_aligned[i] > 25 and 
                di_plus_aligned[i] > di_minus_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Bearish: price below EMA20 + ADX strong + DI- > DI+ + volume
            elif (close[i] < ema_20[i] and 
                  adx_aligned[i] > 25 and 
                  di_minus_aligned[i] > di_plus_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Bands squeeze breakout with 1d trend filter and volume confirmation.
# Long when BB width < 20th percentile + price > upper band + 1d close > 1d SMA50 + volume > 2x avg.
# Short when BB width < 20th percentile + price < lower band + 1d close < 1d SMA50 + volume > 2x avg.
# Exit when price crosses middle band or BB width > 80th percentile (squeeze ends).
# Uses 1h timeframe targeting 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear: squeeze captures low volatility breakouts, 1d filter avoids counter-trend.

name = "1h_bb_squeeze_breakout_1d_trend_vol_v1"
timeframe = "1h"
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
    
    # Bollinger Bands (20, 2) on 1h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = bb_upper - bb_lower
    bb_middle = sma_20
    
    # BB width percentile lookback (50 periods)
    bb_width_percentile = np.zeros_like(bb_width)
    for i in range(50, n):
        window = bb_width[i-50:i]
        if len(window) > 0:
            bb_width_percentile[i] = (np.sum(window < bb_width[i]) / len(window)) * 100
        else:
            bb_width_percentile[i] = 50
    
    # SMA50 (1d trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after BB width percentile warmup
        # Skip if required data not available
        if (np.isnan(bb_width_percentile[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_middle[i]) or np.isnan(sma_50_1d_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses middle band or squeeze ends (BB width > 80th percentile)
        if position == 1:  # long position
            if close[i] <= bb_middle[i] or bb_width_percentile[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] >= bb_middle[i] or bb_width_percentile[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for squeeze breakout entries with trend and volume confirmation
            # Bullish squeeze breakout: BB width < 20th percentile + price > upper band + 1d uptrend + volume spike
            if (bb_width_percentile[i] < 20 and 
                close[i] > bb_upper[i] and 
                close[i] > sma_50_1d_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Bearish squeeze breakout: BB width < 20th percentile + price < lower band + 1d downtrend + volume spike
            elif (bb_width_percentile[i] < 20 and 
                  close[i] < bb_lower[i] and 
                  close[i] < sma_50_1d_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R mean reversion with 4h EMA trend filter and volume confirmation.
# Long when Williams %R < -80 + price > 4h EMA50 + volume > 1.5x avg.
# Short when Williams %R > -20 + price < 4h EMA50 + volume > 1.5x avg.
# Exit when Williams %R crosses -50 (mean) or price crosses 4h EMA50.
# Uses 1h timeframe targeting 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear: Williams %R mean reversion in ranges, EMA50 filter avoids counter-trend.

name = "1h_williams_r_mean_reversion_4h_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 14:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # EMA50 (4h trend filter)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R crosses -50 or price crosses 4h EMA50
        if position == 1:  # long position
            if williams_r[i] >= -50 or close[i] <= ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if williams_r[i] <= -50 or close[i] >= ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for mean reversion entries with trend and volume confirmation
            # Bullish: Williams %R oversold + 4h uptrend + volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_4h_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Bearish: Williams %R overbought + 4h downtrend + volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_4h_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Stochastic Oscillator mean reversion with 4h ADX trend filter and volume confirmation.
# Long when Stoch %K < 20 + %D < 20 + ADX > 25 + DI+ > DI- + volume > 1.5x avg.
# Short when Stoch %K > 80 + %D > 80 + ADX > 25 + DI- > DI+ + volume > 1.5x avg.
# Exit when Stoch %K crosses 50 or ADX < 20 or DI crossover reverses.
# Uses 1h timeframe targeting 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear: Stoch mean reversion in ranges, ADX filters ranging markets.

name = "1h_stochastic_mean_reversion_4h_adx_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 14:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Stochastic Oscillator (14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    stoch_k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    stoch_d = pd.Series(stoch_k_raw).rolling(window=3, min_periods=3).mean().values
    stoch_k = pd.Series(stoch_d).rolling(window=3, min_periods=3).mean().values  # %K smoothed
    
    # ADX (4h trend filter)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr = WilderSmooth(tr, period)
    dm_plus_smooth = WilderSmooth(dm_plus, period)
    dm_minus_smooth = WilderSmooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = WilderSmooth(dx, period)
    
    # Align ADX, DI+, DI- to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    di_plus_aligned = align_htf_to_ltf(prices, df_4h, di_plus)
    di_minus_aligned = align_htf_to_ltf(prices, df_4h, di_minus)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Stochastic warmup
        # Skip if required data not available
        if (np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(di_plus_aligned[i]) or np.isnan(di_minus_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Stoch %K crosses 50 or ADX < 20 or DI crossover reverses
        if position == 1:  # long position
            if stoch_k[i] >= 50 or adx_aligned[i] < 20 or di_minus_aligned[i] > di_plus_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if stoch_k[i] <= 50 or adx_aligned[i] < 20 or di_plus_aligned[i] > di_minus_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for mean reversion entries with trend and volume confirmation
            # Bullish: Stoch oversold + ADX strong + DI+ > DI- + volume spike
            if (stoch_k[i] < 20 and stoch_d[i] < 20 and 
                adx_aligned[i] > 25 and 
                di_plus_aligned[i] > di_minus_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Bearish: Stoch overbought + ADX strong + DI- > DI+ + volume spike
            elif (stoch_k[i] > 80 and stoch_d[i] > 80 and 
                  adx_aligned[i] > 25 and 
                  di_minus_aligned[i] > di_plus_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Commodity Channel Index (CCI) mean reversion with 4h EMA trend filter and volume confirmation.
# Long when CCI < -100 + price > 4h EMA50 + volume > 1.5x avg.
# Short when CCI > +100 + price < 4h EMA50 + volume > 1.5x avg.
# Exit when CCI crosses 0 (mean) or price crosses 4h EMA50.
# Uses 1h timeframe targeting 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear: CCI mean reversion in ranges, EMA50 filter avoids counter-trend.

name = "1h_cci_mean_reversion_4h_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Commodity Channel Index (CCI, 20-period)
    typical_price = (high + low + close) / 3
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mean_deviation = pd.Series(np.abs(typical_price - sma_tp)).rolling(window=20, min_periods=20).mean().values
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-10)
    
    # EMA50 (4h trend filter)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after CCI warmup
        # Skip if required data not available
        if np.isnan(cci[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: CCI crosses 0 or price crosses 4h EMA50
        if position == 1:  # long position
            if cci[i] >= 0 or close[i] <= ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if cci[i] <= 0 or close[i] >= ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for mean reversion entries with trend and volume confirmation
            # Bullish: CCI oversold + 4h uptrend + volume spike
            if (cci[i] < -100 and 
                close[i] > ema_4h_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Bearish: CCI overbought + 4h downtrend + volume spike
            elif (cci[i] > 100 and 
                  close[i] < ema_4h_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Money Flow Index (MFI) mean reversion with 4h ADX trend filter and volume confirmation.
# Long when MFI < 20 + ADX > 25 + DI+ > DI- + volume > 1.5x avg.
# Short when MFI > 80 + ADX > 25 + DI- > DI+ + volume > 1.5x avg.
# Exit when MFI crosses 50 or ADX < 20 or DI crossover reverses.
# Uses 1h timeframe targeting 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear: MFI mean reversion in ranges, ADX filters ranging markets.

name = "1h_mfi_mean_reversion_4h_adx_vol_v1"
timeframe = "1h"
leverage =