# 12h_1d_camarilla_breakout_v3
# Hypothesis: Use 12h candles with daily Camarilla levels for mean-reversion in range-bound markets.
# In 2025 BTC/ETH are expected to trade in ranges, not strong trends.
# Buys near daily L3 (support) and sells near daily H3 (resistance) on 12h timeframe.
# Uses 12h RSI(14) < 30 for long entry and > 70 for short entry to avoid chasing momentum.
# Volume confirmation: 12h volume > 1.5x 20-period average to confirm interest at levels.
# Target: 15-35 trades/year to minimize fee drag in ranging markets.

name = "12h_1d_camarilla_breakout_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    
    # Align to 12h timeframe (daily levels update only after daily bar closes)
    h3_level = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # RSI(14) for momentum filter
    def rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        gain = np.zeros_like(close_prices)
        loss = np.zeros_like(close_prices)
        if len(close_prices) < period:
            return np.full_like(close_prices, 50.0)
        gain[period] = np.mean(up[:period])
        loss[period] = np.mean(down[:period])
        for i in range(period+1, len(close_prices)):
            gain[i] = (gain[i-1] * (period-1) + up[i-1]) / period
            loss[i] = (loss[i-1] * (period-1) + down[i-1]) / period
        rs = np.where(loss != 0, gain / loss, 100)
        rsi_vals = 100 - (100 / (1 + rs))
        rsi_vals[:period] = 50.0
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    rsi_oversold = rsi_vals < 30
    rsi_overbought = rsi_vals > 70
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if levels not ready
        if np.isnan(h3_level[i]) or np.isnan(l3_level[i]):
            signals[i] = 0.0
            continue
        
        # Long setup: near L3 support with oversold RSI and volume
        if (close[i] <= l3_level[i] * 1.02) and rsi_oversold[i] and vol_confirm[i]:
            if position != 1:
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25
        # Short setup: near H3 resistance with overbought RSI and volume
        elif (close[i] >= h3_level[i] * 0.98) and rsi_overbought[i] and vol_confirm[i]:
            if position != -1:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25
        # Exit: price moves back toward middle of range
        elif abs(close[i] - (h3_level[i] + l3_level[i])/2) < (h3_level[i] - l3_level[i]) * 0.1:
            if position == 1:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals