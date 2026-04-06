#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter with 1w Donchian(20) breakout and volume confirmation
# In trending markets (CHOP < 38.2): trade Donchian breakouts (momentum)
# In ranging markets (CHOP > 61.8): mean revert at Donchian bands (reversal)
# Uses 1w Donchian channels for structure, 1d for execution, 1w volume for confirmation
# Position size: 0.25 (25% of capital) with ATR stoploss
# Target: 50-100 total trades over 4 years (12-25/year)

name = "1d_chop_regime_donchian_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for Donchian channels and Choppiness Index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w Donchian(20) channels
    high_series = pd.Series(high_1w)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_1w)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # 1w Choppiness Index (14-period)
    def calculate_chop(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = tr1[0]
        tr3[0] = tr1[0]
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        range_max_min = max_high - min_low
        # Avoid division by zero
        range_max_min = np.where(range_max_min == 0, 1e-10, range_max_min)
        chop = 100 * np.log10(atr / range_max_min) / np.log10(period)
        return chop
    
    chop = calculate_chop(high_1w, low_1w, close_1w, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # 1w volume for confirmation
    volume_1w = df_1w['volume'].values
    volume_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_1w)
    
    # ATR(14) for stoploss (calculated on 1d)
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_1w_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions based on regime
            elif chop_aligned[i] < 38.2:  # trending - exit on opposite breakout
                if close[i] < lower_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            else:  # ranging - exit at opposite band or midline
                if close[i] > (upper_aligned[i] + lower_aligned[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions based on regime
            elif chop_aligned[i] < 38.2:  # trending - exit on opposite breakout
                if close[i] > upper_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            else:  # ranging - exit at opposite band or midline
                if close[i] < (upper_aligned[i] + lower_aligned[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on regime
            # Trending regime (CHOP < 38.2): trade breakouts
            if chop_aligned[i] < 38.2:
                # Long: break above upper band with volume confirmation
                if (close[i] > upper_aligned[i] and
                    volume[i] > 1.5 * volume_ma_1w_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: break below lower band with volume confirmation
                elif (close[i] < lower_aligned[i] and
                      volume[i] > 1.5 * volume_ma_1w_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            # Ranging regime (CHOP > 61.8): mean revert at bands
            elif chop_aligned[i] > 61.8:
                # Long: bounce from lower band
                if (close[i] < lower_aligned[i] * 1.001 and  # near lower band
                    close[i] > lower_aligned[i] and
                    volume[i] > 1.2 * volume_ma_1w_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: rejection at upper band
                elif (close[i] > upper_aligned[i] * 0.999 and  # near upper band
                      close[i] < upper_aligned[i] and
                      volume[i] > 1.2 * volume_ma_1w_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w EMA trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions
# In bullish trend (price > 1w EMA50): buy oversold (%R < -80), take profit at overbought (%R > -20)
# In bearish trend (price < 1w EMA50): sell overbought (%R > -20), take profit at oversold (%R < -80)
# Uses volume spike confirmation to filter false signals
# Position size: 0.25 with ATR stoploss
# Target: 60-120 total trades over 4 years (15-30/year)

name = "1d_williamsr_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1w volume for confirmation
    volume_1w = df_1w['volume'].values
    volume_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_1w)
    
    # Williams %R (14-period) on 1d
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        # Avoid division by zero
        denominator = highest_high - lowest_low
        denominator = np.where(denominator == 0, 1e-10, denominator)
        williams_r = -100 * (highest_high - close) / denominator
        return williams_r
    
    williams_r = calculate_williams_r(high, low, close, 14)
    
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma_1w_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Take profit: Williams %R returns from overbought
            elif williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Take profit: Williams %R returns from oversold
            elif williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Bullish trend: price above 1w EMA50
            if close[i] > ema_1w_aligned[i]:
                # Buy oversold: Williams %R < -80 with volume confirmation
                if (williams_r[i] < -80 and
                    volume[i] > 1.8 * volume_ma_1w_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
            # Bearish trend: price below 1w EMA50
            elif close[i] < ema_1w_aligned[i]:
                # Sell overbought: Williams %R > -20 with volume confirmation
                if (williams_r[i] > -20 and
                    volume[i] > 1.8 * volume_ma_1w_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d ADX trend strength with 1w Parabolic SAR and volume confirmation
# ADX > 25 indicates strong trend - trade in direction of SAR
# ADX < 20 indicates ranging market - fade extreme price action
# Uses 1w Parabolic SAR for trend direction, 1d ADX for regime, 1w volume for confirmation
# Position size: 0.25 with ATR stoploss
# Target: 40-80 total trades over 4 years (10-20/year)

name = "1d_adx_1w_sar_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for Parabolic SAR
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w Parabolic SAR
    def calculate_parabolic_sar(high, low, close, af_start=0.02, af_increment=0.02, af_max=0.2):
        n = len(close)
        sar = np.zeros(n)
        trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
        af = af_start
        
        # Initialize
        if high[1] > high[0]:
            trend[0] = 1
            sar[0] = low[0]
            ep = high[0]
        else:
            trend[0] = -1
            sar[0] = high[0]
            ep = low[0]
        
        for i in range(1, n):
            if trend[i-1] == 1:  # uptrend
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                # SAR cannot exceed low of previous two periods
                sar[i] = min(sar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
                if low[i] > sar[i]:  # trend continues
                    trend[i] = 1
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_increment, af_max)
                else:  # trend reverses to downtrend
                    trend[i] = -1
                    sar[i] = ep
                    ep = low[i]
                    af = af_start
            else:  # downtrend
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                # SAR cannot be below high of previous two periods
                sar[i] = max(sar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
                if high[i] < sar[i]:  # trend continues
                    trend[i] = -1
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_increment, af_max)
                else:  # trend reverses to uptrend
                    trend[i] = 1
                    sar[i] = ep
                    ep = high[i]
                    af = af_start
        
        return sar
    
    sar_1w = calculate_parabolic_sar(high_1w, low_1w, close_1w)
    sar_aligned = align_htf_to_ltf(prices, df_1w, sar_1w)
    
    # 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = tr1[0]
        tr3[0] = tr1[0]
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        tr_ma = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        dm_plus_ma = pd.Series(dm_plus).rolling(window=period, min_periods=period).mean().values
        dm_minus_ma = pd.Series(dm_minus).rolling(window=period, min_periods=period).mean().values
        
        # Directional Indicators
        # Avoid division by zero
        tr_ma_safe = np.where(tr_ma == 0, 1e-10, tr_ma)
        di_plus = 100 * dm_plus_ma / tr_ma_safe
        di_minus = 100 * dm_minus_ma / tr_ma_safe
        
        # DX and ADX
        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        dx = np.where((di_plus + di_minus) == 0, 0, dx)
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # 1w volume for confirmation
    volume_1w = df_1w['volume'].values
    volume_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_1w)
    
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(sar_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(volume_ma_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend weakening or SAR flip
            elif adx[i] < 20 or close[i] < sar_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend weakening or SAR flip
            elif adx[i] < 20 or close[i] > sar_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on ADX regime and SAR
            # Strong trend (ADX > 25): follow SAR
            if adx[i] > 25:
                # Long: SAR below price (uptrend)
                if close[i] > sar_aligned[i]:
                    # Volume confirmation
                    if volume[i] > 1.5 * volume_ma_1w_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                # Short: SAR above price (downtrend)
                elif close[i] < sar_aligned[i]:
                    if volume[i] > 1.5 * volume_ma_1w_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
            # Ranging market (ADX < 20): fade extremes
            elif adx[i] < 20:
                # Long: price near SAR in downtrend context (mean reversion)
                if close[i] < sar_aligned[i] * 1.01 and close[i] > sar_aligned[i]:
                    if volume[i] > 1.2 * volume_ma_1w_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                # Short: price near SAR in uptrend context
                elif close[i] > sar_aligned[i] * 0.99 and close[i] < sar_aligned[i]:
                    if volume[i] > 1.2 * volume_ma_1w_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Stochastic Oscillator with 1w EMA trend filter and volume confirmation
# Stochastic identifies overbought/oversold with momentum
# In bullish trend (price > 1w EMA50): buy oversold stochastic (%K < 20)
# In bearish trend (price < 1w EMA50): sell overbought stochastic (%K > 80)
# Uses volume spike and stochastic crossover for entry confirmation
# Position size: 0.25 with ATR stoploss
# Target: 50-100 total trades over 4 years (12-25/year)

name = "1d_stochastic_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1w volume for confirmation
    volume_1w = df_1w['volume'].values
    volume_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_1w)
    
    # Stochastic Oscillator (14,3,3) on 1d
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        lowest_low = pd.Series(low).rolling(window=k_period, min_periods=k_period).min().values
        highest_high = pd.Series(high).rolling(window=k_period, min_periods=k_period).max().values
        # Avoid division by zero
        denominator = highest_high - lowest_low
        denominator = np.where(denominator == 0, 1e-10, denominator)
        k_percent = 100 * (close - lowest_low) / denominator
        # Smooth %K to get %D
        d_percent = pd.Series(k_percent).rolling(window=d_period, min_periods=d_period).mean().values
        return k_percent, d_percent
    
    stoch_k, stoch_d = calculate_stochastic(high, low, close, 14, 3)
    
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma_1w_aligned[i]) or 
            np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Take profit: stochastic returns from overbought or bearish crossover
            elif stoch_k[i] > 80 or (stoch_k[i] < stoch_d[i] and stoch_k[i] > 50):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Take profit: stochastic returns from oversold or bullish crossover
            elif stoch_k[i] < 20 or (stoch_k[i] > stoch_d[i] and stoch_k[i] < 50):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Bullish trend: price above 1w EMA50
            if close[i] > ema_1w_aligned[i]:
                # Buy oversold: stochastic %K < 20 with volume confirmation
                if (stoch_k[i] < 20 and
                    volume[i] > 1.8 * volume_ma_1w_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
            # Bearish trend: price below 1w EMA50
            elif close[i] < ema_1w_aligned[i]:
                # Sell overbought: stochastic %K > 80 with volume confirmation
                if (stoch_k[i] > 80 and
                    volume[i] > 1.8 * volume_ma_1w_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Commodity Channel Index (CCI) with 1w ADX trend filter and volume confirmation
# CCI > 100 indicates overbought, CCI < -100 indicates oversold
# In strong trend (ADX > 25): fade extremes (mean reversion)
# In weak trend (ADX < 20): trend follow (momentum)
# Uses 1w ADX for regime detection, 1d CCI for signals, 1w volume for confirmation
# Position size: 0.25 with ATR stoploss
# Target: 40-80 total trades over 4 years (10-20/year)

name = "1d_cci_1w_adx_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = tr1[0]
        tr3[0] = tr1[0]
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1)