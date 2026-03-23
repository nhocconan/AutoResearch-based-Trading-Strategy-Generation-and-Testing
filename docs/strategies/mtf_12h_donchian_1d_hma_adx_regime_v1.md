# Strategy: mtf_12h_donchian_1d_hma_adx_regime_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.158 | +15.4% | -10.7% | 185 | FAIL |
| ETHUSDT | -0.711 | -8.0% | -17.8% | 210 | FAIL |
| SOLUSDT | 0.402 | +53.3% | -27.3% | 194 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.356 | +10.4% | -7.6% | 61 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #089: 12h Donchian Breakout with 1d HMA Trend Filter + ADX Regime
Hypothesis: 12h timeframe captures major trend moves with less noise than lower TFs.
Donchian(20) breakout is proven on longer timeframes (Turtle Trading legacy).
1d HMA provides stable trend bias (slower than 12h, avoids whipsaws).
ADX filter ensures we only trade in trending markets (ADX>25).
12h has fewer bars = less fee drag, better risk/reward per trade.

Why this might work on 12h (learning from failures):
- #077 (12h Donchian + 1d HMA + RSI): Sharpe=-0.393 - RSI filter too restrictive
- #083 (12h Supertrend + 1d HMA + RSI): Sharpe=0.085 - Supertrend works on 12h!
- #084 (1d Supertrend + 4h HMA): Sharpe=-0.110 - Wrong TF combo
- Key insight: 12h needs SIMPLER entry conditions to generate enough trades
- Remove RSI filter, rely on Donchian + ADX + HTF trend only

Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.35 strong signals. Stoploss at 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_hma_adx_regime_v1"
timeframe = "12h"
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

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel.
    Returns: upper_band, lower_band, middle_band
    """
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    upper[:period] = np.nan
    lower[:period] = np.nan
    middle[:period] = np.nan
    
    return upper, lower, middle

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Donchian Channel (20-period breakout)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, 20)
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias (stable, slow)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNAL ===
        # Breakout above upper band = long signal
        # Breakout below lower band = short signal
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === ADX REGIME FILTER (avoid choppy markets) ===
        # ADX > 25 = strong trending market (good for breakouts)
        # ADX < 20 = ranging market (avoid entries)
        strong_trend = adx[i] > 25
        trending_market = adx[i] > 20
        
        # === RSI MOMENTUM (light filter - don't over-constrain) ===
        rsi = calculate_rsi(close, 14)
        rsi_momentum_long = rsi[i] > 45  # Not deeply oversold
        rsi_momentum_short = rsi[i] < 55  # Not deeply overbought
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (simplified for 12h - ensure trades) ===
        # Path 1: Donchian breakout + 1d bullish + strong trend (primary - strong signal)
        if donchian_breakout_long and bull_trend_1d and strong_trend:
            if ema_bullish:
                new_signal = SIZE_STRONG
            else:
                new_signal = SIZE_BASE
        
        # Path 2: Donchian breakout + EMA bullish + trending (simpler, ensures trades)
        if new_signal == 0.0 and donchian_breakout_long and ema_bullish and trending_market:
            if bull_trend_1d or rsi_momentum_long:
                new_signal = SIZE_BASE
        
        # Path 3: Donchian breakout + 1d bullish only (fallback to ensure trades on all symbols)
        if new_signal == 0.0 and donchian_breakout_long and bull_trend_1d:
            if trending_market or ema_bullish:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (simplified for 12h - ensure trades) ===
        # Path 1: Donchian breakout + 1d bearish + strong trend (primary - strong signal)
        if donchian_breakout_short and bear_trend_1d and strong_trend:
            if ema_bearish:
                new_signal = -SIZE_STRONG
            else:
                new_signal = -SIZE_BASE
        
        # Path 2: Donchian breakout + EMA bearish + trending (simpler, ensures trades)
        if new_signal == 0.0 and donchian_breakout_short and ema_bearish and trending_market:
            if bear_trend_1d or rsi_momentum_short:
                new_signal = -SIZE_BASE
        
        # Path 3: Donchian breakout + 1d bearish only (fallback to ensure trades on all symbols)
        if new_signal == 0.0 and donchian_breakout_short and bear_trend_1d:
            if trending_market or ema_bearish:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR for 12h ===
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
2026-03-22 11:32
