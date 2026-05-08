# 6h_Equilibrium_Pulse
# Strategy: 6h equilibrium detection using 1d VWAP deviation + volume surge + 1w trend filter
# Long when price > 1d VWAP with volume spike in uptrend, short when price < 1d VWAP with volume spike in downtrend
# Exit when price returns to VWAP or trend weakens
# Target: 60-120 trades over 4 years (15-30/year) to minimize fee drag
# Works in bull (follow 1w trend) and bear (mean revert to VWAP in range)

name = "6h_Equilibrium_Pulse"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d VWAP (typical price * volume) / cumulative volume
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    
    # Calculate 1d VWAP standard deviation for bands
    vwap_dev = typical_price - vwap_values
    vwap_var = (vwap_dev ** 2 * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_std = np.sqrt(np.maximum(vwap_var, 0))
    
    # 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    trend_1w_up = ema_21_1w > np.roll(ema_21_1w, 1)
    trend_1w_up = np.where(np.isnan(trend_21_1w), False, trend_1w_up)
    
    # Align 1d VWAP and bands to 6h
    vwap_6h = align_htf_to_ltf(prices, df_1d, vwap_values)
    vwap_upper_6h = align_htf_to_ltf(prices, df_1d, vwap_values + vwap_std)
    vwap_lower_6h = align_htf_to_ltf(prices, df_1d, vwap_values - vwap_std)
    
    # Align 1w trend to 6h
    trend_1w_up_6h = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    
    # Volume surge detection: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_6h[i]) or np.isnan(vwap_upper_6h[i]) or np.isnan(vwap_lower_6h[i]) or
            np.isnan(trend_1w_up_6h[i]) or np.isnan(volume_surge[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for volume surge away from VWAP in direction of 1w trend
            if volume_surge[i]:
                # Long conditions: price above VWAP, 1w uptrend
                if close[i] > vwap_6h[i] and trend_1w_up_6h[i]:
                    signals[i] = 0.25
                    position = 1
                # Short conditions: price below VWAP, 1w downtrend
                elif close[i] < vwap_6h[i] and not trend_1w_up_6h[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to VWAP or trend weakens
            if close[i] <= vwap_6h[i] or not trend_1w_up_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to VWAP or trend weakens
            if close[i] >= vwap_6h[i] or trend_1w_up_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals