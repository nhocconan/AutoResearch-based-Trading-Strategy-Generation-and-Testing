# Strategy: mtf_1d_supertrend_1w_hma_rsi_pullback_adx_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.103 | +9.7% | -23.5% | 113 | FAIL |
| ETHUSDT | -0.323 | -9.7% | -29.6% | 125 | FAIL |
| SOLUSDT | 1.043 | +237.4% | -35.9% | 121 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.156 | +7.9% | -12.5% | 48 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #096: 1d Supertrend + 1w HMA Trend Filter + RSI Pullback + ADX Regime
Hypothesis: Daily timeframe captures major crypto trends with minimal noise.
1w HMA provides ultra-stable trend bias (slower than 1d, avoids major whipsaws).
Supertrend(10,3) is proven trend-following indicator that works on longer TFs.
RSI pullback entries (not extremes) ensure we enter on dips in uptrends.
ADX filter ensures we only trade in trending markets (ADX>20, lower than usual for 1d).

Why this might work on 1d (learning from #090 Sharpe=0.212):
- #090 had good returns but Sharpe could be improved
- Key insight: 1d needs SIMPLER entry conditions to generate enough trades
- Lower ADX threshold (20 vs 25) to ensure trades on all symbols
- RSI pullback (40-60 range) instead of extremes to catch more entries
- Better stoploss logic with trailing ATR
- Discrete position sizing to minimize fee churn

Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.35 strong signals. Stoploss at 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_supertrend_1w_hma_rsi_pullback_adx_v2"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Calculate TR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth using Wilder's method (EMA with span=period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = long (price above ST), -1 = short (price below ST)
    
    # Calculate basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if np.isnan(atr[i]):
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            continue
            
        # Initial supertrend
        if direction[i-1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            if close[i] < supertrend[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            if close[i] > supertrend[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
    
    return supertrend, direction

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Supertrend (10, 3) - proven parameters
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = higher timeframe trend bias (ultra-stable, very slow)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === SUPERTREND SIGNAL ===
        # st_direction = 1 means price above supertrend (bullish)
        # st_direction = -1 means price below supertrend (bearish)
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === ADX REGIME FILTER (avoid choppy markets) ===
        # ADX > 20 = trending market (lower threshold for 1d to ensure trades)
        # ADX > 25 = strong trending market
        trending_market = adx[i] > 20
        strong_trend = adx[i] > 25
        
        # === RSI PULLBACK (not extremes - catch more entries) ===
        # For longs: RSI 40-60 (pullback in uptrend, not oversold)
        # For shorts: RSI 40-60 (pullback in downtrend, not overbought)
        rsi_pullback_long = 40 <= rsi[i] <= 65
        rsi_pullback_short = 35 <= rsi[i] <= 60
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (simplified for 1d - ensure trades) ===
        # Path 1: Supertrend bullish + 1w bullish + strong trend (primary - strong signal)
        if st_bullish and bull_trend_1w and strong_trend:
            if ema_bullish and rsi_momentum_long:
                new_signal = SIZE_STRONG
            elif ema_bullish or rsi_pullback_long:
                new_signal = SIZE_BASE
        
        # Path 2: Supertrend bullish + EMA bullish + trending (simpler, ensures trades)
        if new_signal == 0.0 and st_bullish and ema_bullish and trending_market:
            if bull_trend_1w or rsi_pullback_long:
                new_signal = SIZE_BASE
        
        # Path 3: Supertrend bullish + 1w bullish only (fallback to ensure trades on all symbols)
        if new_signal == 0.0 and st_bullish and bull_trend_1w:
            if trending_market or ema_bullish:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (simplified for 1d - ensure trades) ===
        # Path 1: Supertrend bearish + 1w bearish + strong trend (primary - strong signal)
        if st_bearish and bear_trend_1w and strong_trend:
            if ema_bearish and rsi_momentum_short:
                new_signal = -SIZE_STRONG
            elif ema_bearish or rsi_pullback_short:
                new_signal = -SIZE_BASE
        
        # Path 2: Supertrend bearish + EMA bearish + trending (simpler, ensures trades)
        if new_signal == 0.0 and st_bearish and ema_bearish and trending_market:
            if bear_trend_1w or rsi_pullback_short:
                new_signal = -SIZE_BASE
        
        # Path 3: Supertrend bearish + 1w bearish only (fallback to ensure trades on all symbols)
        if new_signal == 0.0 and st_bearish and bear_trend_1w:
            if trending_market or ema_bearish:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR for 1d ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 11:38
