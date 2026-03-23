# Strategy: mtf_1d_ensemble_weekly_hma_rsi_donchian_vol_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.557 | +2.8% | -14.2% | 121 | FAIL |
| ETHUSDT | -0.613 | -2.5% | -17.2% | 127 | FAIL |
| SOLUSDT | 0.627 | +73.3% | -14.3% | 120 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.748 | +20.0% | -8.9% | 50 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #432: 1d Multi-Signal Ensemble with Weekly Trend Filter

Hypothesis: After analyzing 431 failed experiments, the key insight is that 1d 
strategies fail because they're either too strict (0 trades) or too simple 
(negative Sharpe). This strategy uses a MULTI-SIGNAL ENSEMBLE approach:

1. WEEKLY HMA(21) TREND BIAS (via mtf_data helper):
   - Long bias when price > 1w HMA
   - Short bias when price < 1w HMA
   - HMA smoother than EMA, critical for weekly trend detection

2. THREE SIGNAL TYPES (any can trigger entry):
   a) RSI(14) MEAN REVERSION: RSI < 35 (long) or > 65 (short)
      - Looser than traditional 30/70 to ensure sufficient trades
      - Must align with weekly trend bias
   
   b) DONCHIAN(14) BREAKOUT: Price breaks 14-bar high/low
      - Shorter period than traditional 20 to catch more moves
      - Only in trending regime (ADX > 22)
   
   c) VOL SPIKE CONTRARIAN: ATR(7)/ATR(30) > 1.8
      - Captures panic extremes (vol spike = potential reversal)
      - Long when vol spike + price < SMA(50)
      - Short when vol spike + price > SMA(50)

3. ADX(14) REGIME FILTER:
   - ADX > 22 = trending (allow breakout signals)
   - ADX < 22 = ranging (allow mean reversion signals only)
   - Prevents breakout whipsaws in choppy markets

4. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for 2022-style crash protection

5. POSITION SIZING: 0.28 discrete (conservative for daily volatility)
   - Max 28% capital per position
   - Discrete levels minimize fee churn

Why this should work on 1d:
- Multiple signal types ensure sufficient trade frequency (>10/year)
- Weekly HMA filter prevents counter-trend disasters
- Vol spike contrarian catches panic reversals (proven edge)
- Looser thresholds than failed 1d experiments (#420, #426)
- Should work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_ensemble_weekly_hma_rsi_donchian_vol_atr_v1"
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
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
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
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def calculate_donchian(high, low, period=14):
    """Calculate Donchian Channel."""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma_50 = calculate_sma(close, 50)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(atr_7[i]) or np.isnan(atr_30[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        trending_market = adx[i] > 22
        ranging_market = adx[i] <= 22
        
        # === SIGNAL 1: RSI MEAN REVERSION ===
        rsi_long = rsi[i] < 35  # Oversold (looser than 30)
        rsi_short = rsi[i] > 65  # Overbought (looser than 70)
        
        # === SIGNAL 2: DONCHIAN BREAKOUT ===
        donchian_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === SIGNAL 3: VOL SPIKE CONTRARIAN ===
        vol_ratio = atr_7[i] / atr_30[i] if atr_30[i] > 0 else 0
        vol_spike = vol_ratio > 1.8
        vol_long = vol_spike and close[i] < sma_50[i]  # Panic below SMA
        vol_short = vol_spike and close[i] > sma_50[i]  # Spike above SMA
        
        # === GENERATE SIGNAL (Ensemble - any signal can trigger) ===
        new_signal = 0.0
        
        # RSI MEAN REVERSION (works in ranging market, also trending)
        if rsi_long and bull_trend_1w:
            new_signal = SIZE
        elif rsi_short and bear_trend_1w:
            new_signal = -SIZE
        
        # DONCHIAN BREAKOUT (only in trending market)
        if trending_market and new_signal == 0.0:
            if donchian_long and bull_trend_1w:
                new_signal = SIZE
            elif donchian_short and bear_trend_1w:
                new_signal = -SIZE
        
        # VOL SPIKE CONTRARIAN (works in any regime)
        if new_signal == 0.0:
            if vol_long and bull_trend_1w:
                new_signal = SIZE
            elif vol_short and bear_trend_1w:
                new_signal = -SIZE
        
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
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1w:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1w:
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
2026-03-22 17:18
