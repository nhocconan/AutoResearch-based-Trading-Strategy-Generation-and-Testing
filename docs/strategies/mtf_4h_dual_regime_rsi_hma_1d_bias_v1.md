# Strategy: mtf_4h_dual_regime_rsi_hma_1d_bias_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.227 | +9.1% | -13.9% | 358 | FAIL |
| ETHUSDT | 0.417 | +46.8% | -12.9% | 376 | PASS |
| SOLUSDT | 0.421 | +59.4% | -39.6% | 369 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.741 | +17.3% | -10.0% | 112 | PASS |
| SOLUSDT | 0.103 | +6.9% | -15.3% | 122 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #349: 4h Primary + 1d HTF — Dual Regime with RSI Mean Reversion + HMA Trend

Hypothesis: Previous 4h strategies failed because:
1. Too many filters (CRSI + Choppiness + Donchian + Volume all required)
2. CRSI calculation is complex and may not add value over simpler RSI
3. Symmetric thresholds don't adapt to bull/bear regimes

This strategy uses SIMPLER but MORE ROBUST logic:
1. 1d HMA(21) as MACRO BIAS (only long if price > 1d HMA, only short if price < 1d HMA)
2. 4h Choppiness Index for regime detection (CHOP>55=range, CHOP<45=trend)
3. RANGE REGIME: RSI(14) extremes for mean reversion (RSI<30 long, RSI>70 short)
4. TREND REGIME: 4h HMA(16/48) crossover + ADX(14)>25 for trend confirmation
5. ATR(14) trailing stop at 2.5x for risk management
6. Relaxed thresholds to ensure 20-50 trades/year on 4h

KEY INSIGHT: Simpler RSI(14) with proper regime filter outperforms complex CRSI.
The 1d HMA bias prevents counter-trend trades that whipsaw in crashes.
Dual regime (mean revert in chop, trend follow otherwise) adapts to market conditions.

TARGET: 25-50 trades/year on 4h, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_rsi_hma_1d_bias_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # HMA for trend detection (fast and slow)
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    
    # Calculate and align 1d HMA for macro bias (HARD FILTER)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 4h (target 25-50 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA - HARD FILTER) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # High choppiness = range regime (mean revert)
        is_trending = chop[i] < 45.0  # Low choppiness = trend regime (breakout)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: RSI mean reversion
            # Long: RSI < 32 + price above 1d HMA (bullish macro)
            # Short: RSI > 68 + price below 1d HMA (bearish macro)
            
            if price_above_hma_1d and rsi_14[i] < 32:
                # Long oversold in bullish macro (range regime)
                desired_signal = BASE_SIZE
            
            elif price_below_hma_1d and rsi_14[i] > 68:
                # Short overbought in bearish macro (range regime)
                desired_signal = -BASE_SIZE
        
        elif is_trending:
            # TREND REGIME: HMA crossover + ADX confirmation
            # Long: HMA16 > HMA48 + ADX > 25 + 1d bullish
            # Short: HMA16 < HMA48 + ADX > 25 + 1d bearish
            
            hma_bullish = hma_16[i] > hma_48[i]
            hma_bearish = hma_16[i] < hma_48[i]
            trend_strong = adx_14[i] > 25.0
            
            if price_above_hma_1d and hma_bullish and trend_strong:
                # Long trend in bullish macro (trend regime)
                desired_signal = BASE_SIZE
            
            elif price_below_hma_1d and hma_bearish and trend_strong:
                # Short trend in bearish macro (trend regime)
                desired_signal = -BASE_SIZE
        
        else:
            # NEUTRAL REGIME (45 <= CHOP <= 55): Reduced size, wait for clarity
            # Only take high-conviction RSI extremes
            
            if price_above_hma_1d and rsi_14[i] < 28:
                desired_signal = BASE_SIZE * 0.7
            
            elif price_below_hma_1d and rsi_14[i] > 72:
                desired_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === RSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and rsi_14[i] > 65:
            # Long position: exit when RSI reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 35:
            # Short position: exit when RSI reaches oversold
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if regime and bias still valid
            if position_side > 0:
                if (is_choppy and price_above_hma_1d and rsi_14[i] < 65) or \
                   (is_trending and price_above_hma_1d and hma_16[i] > hma_48[i]):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if (is_choppy and price_below_hma_1d and rsi_14[i] > 35) or \
                   (is_trending and price_below_hma_1d and hma_16[i] < hma_48[i]):
                    desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 08:59
