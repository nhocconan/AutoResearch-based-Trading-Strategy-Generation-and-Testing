# 4h_12h_camarilla_volume_trend_v1
# Hypothesis: 4h trend from 12h EMA200 + Camarilla pivot (1d H4/L4) breakout + volume filter
# Long when: price > 12h EMA200 (trend), breaks above 1d H4, volume > 1.5x 20-bar avg
# Short when: price < 12h EMA200, breaks below 1d L4, volume > 1.5x 20-bar avg
# Exit: price crosses 12h EMA200 (trend reversal) or reverses to 1d pivot point
# Target: 30-60 trades/year to minimize fee drag while capturing trend momentum.
# Works in bull via breakouts with trend, in bear via short breakdowns with trend filter.

name = "4h_12h_camarilla_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    # Calculate 12h EMA200
    close_12h = df_12h['close'].values
    ema_200 = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200)
    
    # Get 1d data for Camarilla pivot levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # first value has no previous day
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot point and Camarilla levels
    pp = (prev_high + prev_low + prev_close) / 3.0
    range_1d = prev_high - prev_low
    
    # Camarilla levels: H4 = PP + 1.1/2 * range, L4 = PP - 1.1/2 * range
    h4 = pp + (1.1 / 2) * range_1d
    l4 = pp - (1.1 / 2) * range_1d
    
    # Align daily levels to 4h timeframe (daily values update after daily bar closes)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume confirmation: volume > 1.5 * 20-period average (4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(ema_200_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(pp_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter from 12h EMA200
        uptrend = close[i] > ema_200_aligned[i]
        downtrend = close[i] < ema_200_aligned[i]
        
        # Long signal: uptrend + price breaks above H4
        if uptrend and close[i] > h4_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: downtrend + price breaks below L4
        elif downtrend and close[i] < l4_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and (close[i] <= ema_200_aligned[i] or close[i] <= pp_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= ema_200_aligned[i] or close[i] >= pp_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals