#130974
# Hypothesis: 1h Donchian breakout (20-period) with 4h trend filter (ADX > 25) and volume confirmation
# Uses 4h ADX to filter for trending markets only, reducing whipsaws in ranging markets
# Donchian breakout captures momentum in trending markets
# Volume confirmation (>1.5x 20-bar average) ensures institutional participation
# Designed for 1h timeframe to target 60-150 total trades over 4 years (15-37/year)
# Works in both bull/bear: captures strong trends, avoids false signals in consolidation
# Uses 4h timeframe for signal direction, 1h only for entry timing

name = "1h_Donchian20_4hADX25_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 35:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ADX(25) trend filter
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr_4h = wilder_smooth(tr, 25)
    dm_plus_smooth = wilder_smooth(dm_plus, 25)
    dm_minus_smooth = wilder_smooth(dm_minus, 25)
    
    di_plus = np.where(atr_4h != 0, 100 * dm_plus_smooth / atr_4h, 0)
    di_minus = np.where(atr_4h != 0, 100 * dm_minus_smooth / atr_4h, 0)
    
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_4h = wilder_smooth(dx, 25)
    
    # Calculate Donchian channels (20-period) on 1h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_4h_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian upper band AND trending market (ADX > 25) AND volume confirmation
            if close[i] > highest_high[i] and adx_4h_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below Donchian lower band AND trending market (ADX > 25) AND volume confirmation
            elif close[i] < lowest_low[i] and adx_4h_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower band
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above Donchian upper band
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals