# 4h_RSI4_BB20_Breakout_BullBear
# Hypothesis: RSI(4) crossing above 60 with Bollinger Band(20,2) upper band breakout for longs,
# and RSI(4) crossing below 40 with lower band breakout for shorts, only in trending regimes (ADX(14) > 25).
# Uses Bollinger Bands for dynamic support/resistance and breakouts, RSI for momentum confirmation,
# and ADX to filter ranging markets. Designed for ~20-30 trades/year on 4h timeframe.
# Works in bull markets via momentum breakouts and in bear markets via mean-reversion failures
# (false breakouts in ranging markets filtered by ADX).
name = "4h_RSI4_BB20_Breakout_BullBear"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_middle = bb_middle.values
    bb_upper = bb_upper.values
    bb_lower = bb_lower.values
    
    # RSI(4)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # ADX(14) for trend filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx = adx.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(rsi[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        rsi_val = rsi[i]
        adx_val = adx[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        
        # Trend filter: only trade when ADX > 25 (trending market)
        if adx_val <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI(4) > 60 and close breaks above BB upper
            if rsi_val > 60 and close_val > bb_upper_val:
                signals[i] = 0.25
                position = 1
            # Short: RSI(4) < 40 and close breaks below BB lower
            elif rsi_val < 40 and close_val < bb_lower_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI < 50 or close below BB middle
            if rsi_val < 50 or close_val < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI > 50 or close above BB middle
            if rsi_val > 50 or close_val > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals