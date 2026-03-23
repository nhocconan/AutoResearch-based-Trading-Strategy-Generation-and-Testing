# Strategy: mtf_12h_adx_regime_1d_hma_donchian_rsi_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.343 | +10.4% | -8.9% | 170 | FAIL |
| ETHUSDT | -0.655 | -4.4% | -15.2% | 200 | FAIL |
| SOLUSDT | 0.280 | +39.0% | -24.4% | 183 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.310 | +9.4% | -6.3% | 59 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #413: 12h ADX Regime + 1d HMA Trend + Donchian/RSI Adaptive Entry

Hypothesis: After 412 failed experiments, the key insight is that 12h timeframe
needs STRONGER regime filtering than 4h. 12h bars are less noisy but require
clearer trend confirmation. This strategy uses:

1. ADX(14) REGIME DETECTION on 12h:
   - ADX > 25 = trending (use Donchian breakout)
   - ADX < 20 = ranging (use RSI mean-reversion)
   - 20-25 = neutral (stay flat, avoid whipsaw)
   - This is MORE strict than 4h strategies to reduce false signals on 12h

2. 1d HMA(21) TREND BIAS (via mtf_data helper):
   - Long only when price > 1d HMA in trending regime
   - Short only when price < 1d HMA in trending regime
   - HMA smoother than EMA, critical for 12h/1d alignment

3. DONCHIAN(20) BREAKOUT for trending regime:
   - Long when price breaks 20-bar high + ADX > 25 + price > 1d HMA
   - Short when price breaks 20-bar low + ADX > 25 + price < 1d HMA
   - Captures sustained moves, not noise

4. RSI(14) MEAN REVERSION for ranging regime:
   - Long when RSI < 30 + price > 1d HMA (bullish dip)
   - Short when RSI > 70 + price < 1d HMA (bearish rally)
   - Only enter with 1d trend bias (avoid counter-trend)

5. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from 2022-style crashes

6. POSITION SIZING: 0.30 discrete (conservative for 12h volatility)
   - Max 30% capital per position
   - Discrete levels minimize fee churn on slower timeframe

Why 12h should work better than 4h:
- Fewer false breakouts (12h closes are more significant)
- Less fee drag (fewer trades, ~20-40/year vs 100+ on 4h)
- Better alignment with 1d HTF (12h = 2 bars per day)
- Should work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_adx_regime_1d_hma_donchian_rsi_atr_v1"
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
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending market
    ADX < 20 = ranging market
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth using Wilder's method (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI and DX
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        # ADX is smoothed DX
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        trending_market = adx[i] > 25
        ranging_market = adx[i] < 20
        # neutral_market = 20 <= ADX <= 25 (stay flat)
        
        # === 1d HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS (for trending regime) ===
        donchian_long = close[i] > donchian_upper[i-1]  # Break above previous high
        donchian_short = close[i] < donchian_lower[i-1]  # Break below previous low
        
        # === RSI MEAN REVERSION SIGNALS (for ranging regime) ===
        rsi_long = rsi[i] < 30  # Oversold
        rsi_short = rsi[i] > 70  # Overbought
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # TRENDING REGIME: Donchian breakout with 1d HMA filter
        if trending_market:
            if bull_trend_1d and donchian_long:
                new_signal = SIZE
            elif bear_trend_1d and donchian_short:
                new_signal = -SIZE
        
        # RANGING REGIME: RSI mean-reversion with 1d HMA filter
        elif ranging_market:
            # Only enter with trend bias (avoid counter-trend mean reversion)
            if bull_trend_1d and rsi_long:
                new_signal = SIZE
            elif bear_trend_1d and rsi_short:
                new_signal = -SIZE
        # NEUTRAL REGIME: Stay flat (20 <= ADX <= 25)
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME FLIP EXIT ===
        # Exit if regime changes against position type
        if in_position and new_signal != 0.0:
            # Long position in trending regime should exit if market becomes ranging without RSI signal
            if position_side > 0 and ranging_market and not rsi_long:
                new_signal = 0.0
            # Short position in trending regime should exit if market becomes ranging without RSI signal
            if position_side < 0 and ranging_market and not rsi_short:
                new_signal = 0.0
        
        # === TREND REVERSAL EXIT (for trending regime positions) ===
        if in_position and new_signal != 0.0 and trending_market:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 17:03
