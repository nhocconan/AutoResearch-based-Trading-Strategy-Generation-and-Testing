# 4h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Use Camarilla R3/S3 levels from 1d as breakout levels, filtered by 1d EMA trend and volume spike.
# In trending markets (1d EMA34 up/down), breakouts above R3 or below S3 with volume > 1.5x average continue the trend.
# Works in bull (breakouts up) and bear (breakouts down). Low trade frequency (~20-50/year) avoids fee drag.
# Uses 1d trend filter to avoid counter-trend breakouts in ranging markets.
# Risk management via signal reversal (no separate stoploss needed).

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (H,L,C)
    # Camarilla: R4 = C + (H-L)*1.5/2, R3 = C + (H-L)*1.25/2, S3 = C - (H-L)*1.25/2
    # We use previous day's H,L,C to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day has no previous, set to NaN
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla R3 and S3
    R3 = prev_close + (prev_high - prev_low) * 1.25 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.25 / 2
    
    # Align Camarilla levels to 4h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_1d > np.roll(ema_34_1d, 1)
    trend_up = np.where(np.isnan(trend_up), False, trend_up)
    trend_down = ema_34_1d < np.roll(ema_34_1d, 1)
    trend_down = np.where(np.isnan(trend_down), False, trend_down)
    
    # Align trend to 4h
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down.astype(float))
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakouts in direction of 1d trend
            if trend_up_aligned[i] and volume_spike[i]:
                # Bullish trend: look for breakout above R3
                if close[i] > R3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif trend_down_aligned[i] and volume_spike[i]:
                # Bearish trend: look for breakdown below S3
                if close[i] < S3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit on breakdown below S3 (contrarian signal)
            if close[i] < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on breakout above R3 (contrarian signal)
            if close[i] > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals