# 4h Donchian Breakout with 1d ADX Trend Filter and Volume Confirmation
# Combines Donchian channel breakouts for trend capture with 1d ADX to ensure
# trades align with higher timeframe trend strength, reducing false signals in chop.
# Volume confirmation filters weak breakouts. Designed for 20-50 trades/year
# to avoid fee drag while maintaining edge in bull/bear regimes.
name = "4h_Donchian20_1dADX_Volume_v2"
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
    
    # 1d ADX for trend strength filter (uses 14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0.0
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0.0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0.0
    
    # Smoothed values
    def smooth_values(values, period):
        smoothed = np.zeros_like(values)
        if len(values) < period:
            return smoothed
        smoothed[period-1] = values[:period].sum()
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    period_adx = 14
    atr_1d = smooth_values(tr_1d, period_adx)
    plus_di_1d = smooth_values(plus_dm, period_adx)
    minus_di_1d = smooth_values(minus_dm, period_adx)
    
    # Avoid division by zero
    dx = np.zeros_like(atr_1d)
    mask = atr_1d > 0
    dx[mask] = (np.abs(plus_di_1d[mask] - minus_di_1d[mask]) / atr_1d[mask]) * 100
    
    adx_1d = smooth_values(dx, period_adx)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian Channel (20-period)
    donchian_len = 20
    upper_channel = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_len, 20, period_adx*2)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Require ADX > 25 for trending market
        if adx_1d_aligned[i] <= 25:
            # In chop, stay flat or maintain position only if already in trend
            if position != 0:
                signals[i] = 0.25 if position == 1 else -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation
            if close[i] > upper_channel[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation
            elif close[i] < lower_channel[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower Donchian
            if close[i] < lower_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper Donchian
            if close[i] > upper_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals