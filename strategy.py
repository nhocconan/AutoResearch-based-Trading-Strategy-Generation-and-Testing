#!/usr/bin/env python3
"""
Experiment #487: 15m Trend-Pullback with 4h HMA Bias

Hypothesis: After 476 failed experiments, the critical insight is that 15m strategies
fail when they're too complex (regime filters, choppiness index, etc.) which reduces
trade count below the minimum threshold. This strategy uses SIMPLE, PROVEN logic:

1. 4H HMA(21) TREND BIAS (via mtf_data helper):
   - Fast enough for 15m entries, slower than 1h to avoid whipsaw
   - Price > 4h HMA = bull bias (favor long pullbacks)
   - Price < 4h HMA = bear bias (favor short rallies)

2. 15M RSI(7) PULLBACK ENTRIES:
   - RSI(7) instead of RSI(14) = faster signals for 15m TF
   - Long: RSI(7) < 35 in bull regime (buy dip)
   - Short: RSI(7) > 65 in bear regime (short rally)
   - Looser thresholds than failed strategies (35/65 vs 38/62)

3. ADX(14) TREND CONFIRMATION:
   - ADX > 20 (not 25+) = ensures trend without killing trade count
   - Critical filter to avoid ranging market whipsaws

4. ATR(14) * 2.5 TRAILING STOP:
   - Tighter than daily strategies (2.5x vs 3.0x) for 15m volatility
   - Signal → 0 when price moves 2.5*ATR against position

5. POSITION SIZING: 0.25 discrete
   - Conservative for 15m volatility (lower than daily 0.30)
   - Discrete levels minimize fee churn

Why this should work on 15m:
- SIMPLE logic = MORE trades (critical after 476 failures with <10 trades)
- 4h HMA provides robust trend bias without weekly slowness
- RSI(7) generates frequent pullback signals on 15m
- ADX > 20 filters chop without being too restrictive
- Should generate 50-100 trades/year per symbol (well above minimum)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_trend_pullback_4h_hma_rsi7_adx_atr_v1"
timeframe = "15m"
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

def calculate_rsi(close, period=7):
    """Calculate Relative Strength Index (fast period for 15m)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    return close_s.ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 7)  # Fast RSI for 15m
    adx = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25  # Conservative for 15m volatility
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_regime = close[i] > hma_4h_aligned[i]
        bear_regime = close[i] < hma_4h_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        trend_confirmed = adx[i] > 20  # Looser than 25 to get more trades
        
        # === SIMPLE PULLBACK ENTRY LOGIC ===
        new_signal = 0.0
        
        # BULL REGIME: Long on RSI pullback
        if bull_regime and trend_confirmed:
            # RSI(7) < 35 = oversold pullback in uptrend
            if rsi[i] < 35:
                new_signal = SIZE
            # Also enter if price pulls back to EMA21 in strong trend
            elif adx[i] > 30 and close[i] < ema_21[i] * 1.005 and close[i] > ema_21[i] * 0.995:
                new_signal = SIZE
        
        # BEAR REGIME: Short on RSI rally
        if bear_regime and trend_confirmed:
            # RSI(7) > 65 = overbought rally in downtrend
            if rsi[i] > 65:
                new_signal = -SIZE
            # Also enter if price rallies to EMA21 in strong trend
            elif adx[i] > 30 and close[i] > ema_21[i] * 0.995 and close[i] < ema_21[i] * 1.005:
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
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime:
                new_signal = 0.0
            if position_side < 0 and bull_regime:
                new_signal = 0.0
        
        # === ADX DROPS BELOW THRESHOLD EXIT ===
        # Exit if trend weakens significantly
        if in_position and adx[i] < 15:
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