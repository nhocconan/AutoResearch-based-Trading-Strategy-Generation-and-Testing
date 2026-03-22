#!/usr/bin/env python3
"""
Experiment #470: 30m Supertrend + RSI Pullback with 4h/1d Regime Filter

Hypothesis: After analyzing 469 failed experiments, the pattern is clear:
- Pure trend strategies fail on BTC/ETH whipsaws (2022 crash destroyed gains)
- Pure mean reversion fails on strong trends (SOL 100x rally)
- 30m timeframe needs STRONGER filtering than 4h/12h due to noise

This strategy combines:
1. 4H HMA(21) TREND BIAS: Smooth trend direction (HMA less lag than EMA)
   - Long bias when price > 4h HMA
   - Short bias when price < 4h HMA
   
2. 1D ADX(14) REGIME FILTER: Detect trending vs ranging markets
   - ADX > 25 = trending (allow Supertrend breakouts)
   - ADX < 25 = ranging (allow RSI mean reversion only)
   - Prevents breakout whipsaws in choppy conditions

3. SUPERTREND(10, 3.0) ENTRY SIGNAL:
   - Clean trend-following signal with ATR-based stops built-in
   - Long when price crosses above Supertrend lower band
   - Short when price crosses below Supertrend upper band
   - Only in trending regime (1d ADX > 25)

4. RSI(14) PULLBACK ENTRY (looser thresholds for more trades):
   - Long: RSI < 40 (not 30) + price > 4h HMA
   - Short: RSI > 60 (not 70) + price < 4h HMA
   - Works in any regime, ensures sufficient trade count

5. ATR(14) TRAILING STOP at 2.0x:
   - Signal → 0 when price moves 2.0*ATR against position
   - Critical for crash protection

6. POSITION SIZING: 0.28 discrete
   - Max 28% capital per position
   - Discrete levels (0.0, ±0.28) minimize fee churn

Why this should work on 30m:
- 4h HMA provides stable trend bias (not noisy like 1h)
- 1d ADX regime filter prevents whipsaw entries
- Supertrend + RSI dual entry ensures >10 trades/year
- Looser RSI thresholds (40/60 vs 30/70) guarantee trades on all symbols
- 2.0x ATR stoploss protects against 2022-style crashes

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_rsi_4h_hma_1d_adx_regime_atr_v1"
timeframe = "30m"
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

def calculate_supertrend(high, low, close, atr, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    trend = np.ones(n)  # 1 = uptrend, -1 = downtrend
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        
        hl2 = (high[i] + low[i]) / 2
        upper_band[i] = hl2 + multiplier * atr[i]
        lower_band[i] = hl2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = -1 if close[i] < upper_band[i] else 1
        else:
            # Upper band logic
            if upper_band[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]:
                upper_band[i] = upper_band[i]
            else:
                upper_band[i] = upper_band[i-1]
            
            # Lower band logic
            if lower_band[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]:
                lower_band[i] = lower_band[i]
            else:
                lower_band[i] = lower_band[i-1]
            
            # Trend determination
            if trend[i-1] == 1:
                if close[i] < lower_band[i]:
                    trend[i] = -1
                    supertrend[i] = upper_band[i]
                else:
                    trend[i] = 1
                    supertrend[i] = lower_band[i]
            else:
                if close[i] > upper_band[i]:
                    trend[i] = 1
                    supertrend[i] = lower_band[i]
                else:
                    trend[i] = -1
                    supertrend[i] = upper_band[i]
    
    return supertrend, trend, upper_band, lower_band

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_trend, st_upper, st_lower = calculate_supertrend(high, low, close, atr, 10, 3.0)
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D ADX REGIME FILTER ===
        adx_1d_val = adx_1d_aligned[i]
        trending_market = adx_1d_val > 25
        ranging_market = adx_1d_val <= 25
        
        # === SUPERTREND SIGNAL ===
        # Long: Supertrend flips to uptrend (trend changes from -1 to 1)
        # Short: Supertrend flips to downtrend (trend changes from 1 to -1)
        supertrend_long = False
        supertrend_short = False
        
        if i > 100 and not np.isnan(st_trend[i-1]):
            if st_trend[i] == 1 and st_trend[i-1] == -1:
                supertrend_long = True
            elif st_trend[i] == -1 and st_trend[i-1] == 1:
                supertrend_short = True
        
        # === RSI PULLBACK SIGNAL (looser thresholds for more trades) ===
        rsi_long = rsi[i] < 40  # Was 30, loosened to ensure trades
        rsi_short = rsi[i] > 60  # Was 70, loosened to ensure trades
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # SUPERTREND BREAKOUT (only in trending market per 1d ADX)
        if trending_market:
            if supertrend_long and bull_trend_4h:
                new_signal = SIZE
            elif supertrend_short and bear_trend_4h:
                new_signal = -SIZE
        
        # RSI PULLBACK (works in any regime, must align with 4h trend)
        if new_signal == 0.0:
            if rsi_long and bull_trend_4h:
                new_signal = SIZE
            elif rsi_short and bear_trend_4h:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
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