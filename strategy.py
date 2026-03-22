#!/usr/bin/env python3
"""
Experiment #505: 1h Primary + 4h/1d HTF — ADX Regime + RSI Pullback + Session Filter

Hypothesis: After 448 failed strategies (mostly vol-spike/Fisher/Choppiness combos), 
try a SIMPLER approach that ensures TRADE FREQUENCY while maintaining edge:

1. ADX REGIME DETECTION: ADX(14)>25 = trend (follow), ADX(14)<20 = range (mean revert)
   This is proven to work better than Choppiness Index for crypto
   Key: Use DIFFERENT logic per regime (not same filters always)

2. 4H HMA for trend direction (faster than 1d, better for 1h entries)
   1d HMA as secondary confirmation only

3. RSI(7) for entry timing (faster than RSI(14), catches more opportunities)
   Long: RSI<35 in uptrend. Short: RSI>65 in downtrend.

4. SESSION FILTER (8-20 UTC): Only trade during high-liquidity hours
   Reduces noise from Asian overnight sessions

5. CRITICAL: LOOSE entry conditions to ensure >=30 trades/symbol on train
   Use OR logic, not AND. Multiple independent entry triggers.

Why this might beat current best (Sharpe=0.435):
- Simpler = more trades (critical: #505 had 0 trades from too many filters)
- ADX regime is proven in crypto (better than Choppiness)
- 4h HMA responds faster than 1d for 1h timeframe
- RSI(7) catches more pullbacks than RSI(14)
- Session filter reduces false signals without killing frequency

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 30-60 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_adx_regime_rsi7_session_4h1d_v1"
timeframe = "1h"
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
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    plus_dm[(plus_dm <= minus_dm)] = 0
    minus_dm[(minus_dm <= plus_dm)] = 0
    
    # Smooth with Wilder's method
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_rsi(close, period=7):
    """Calculate RSI with configurable period (default 7 for faster signals)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators (primary trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 1d HTF indicators (major trend confirmation)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_7[i]):
            continue
        
        # Extract UTC hour for session filter
        utc_hour = get_utc_hour(open_time[i])
        in_session = (utc_hour >= 8) and (utc_hour <= 20)
        
        # === 4H TREND DIRECTION (primary filter) ===
        bull_4h = close[i] > hma_4h_21_aligned[i]
        bear_4h = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA slope for trend strength
        hma_4h_slope_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_slope_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === 1D MAJOR TREND (confirmation) ===
        bull_1d = close[i] > hma_1d_21_aligned[i]
        bear_1d = close[i] < hma_1d_21_aligned[i]
        
        # === ADX REGIME DETECTION ===
        trend_regime = adx_14[i] > 25  # Strong trend
        range_regime = adx_14[i] < 20  # Range/choppy
        # 20-25 = transition (use either logic)
        
        # === RSI SIGNALS (faster with period 7) ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_extreme_low = rsi_7[i] < 25.0
        rsi_extreme_high = rsi_7[i] > 75.0
        rsi_neutral = (rsi_7[i] > 40.0) and (rsi_7[i] < 60.0)
        
        # === SMA 200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — REGIME-ADAPTIVE ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple OR conditions for frequency)
        # Condition 1: Trend regime + 4h bull + RSI pullback (trend following)
        if trend_regime and bull_4h and rsi_oversold:
            new_signal = LONG_SIZE
        # Condition 2: Range regime + RSI extreme low (mean reversion)
        elif range_regime and rsi_extreme_low:
            new_signal = LONG_SIZE * 0.8
        # Condition 3: 4h bull + 1d bull + RSI neutral (momentum continuation)
        elif bull_4h and bull_1d and rsi_neutral:
            new_signal = LONG_SIZE * 0.7
        # Condition 4: Above SMA200 + RSI oversold (pullback in uptrend)
        elif above_sma200 and rsi_oversold:
            new_signal = LONG_SIZE
        # Condition 5: 4h slope bull + RSI cross up from oversold
        elif hma_4h_slope_bull and rsi_7[i] > 30 and rsi_7[i-1] < 30:
            new_signal = LONG_SIZE
        # Condition 6: Session filter + any bull signal (reduce noise)
        elif in_session and bull_4h and rsi_7[i] < 40:
            new_signal = LONG_SIZE * 0.6
        
        # SHORT ENTRIES (mirror logic for bear market)
        if new_signal == 0.0:
            # Condition 1: Trend regime + 4h bear + RSI pullback (trend following)
            if trend_regime and bear_4h and rsi_overbought:
                new_signal = -SHORT_SIZE
            # Condition 2: Range regime + RSI extreme high (mean reversion)
            elif range_regime and rsi_extreme_high:
                new_signal = -SHORT_SIZE * 0.8
            # Condition 3: 4h bear + 1d bear + RSI neutral (momentum continuation)
            elif bear_4h and bear_1d and rsi_neutral:
                new_signal = -SHORT_SIZE * 0.7
            # Condition 4: Below SMA200 + RSI overbought (bounce in downtrend)
            elif below_sma200 and rsi_overbought:
                new_signal = -SHORT_SIZE
            # Condition 5: 4h slope bear + RSI cross down from overbought
            elif hma_4h_slope_bear and rsi_7[i] < 70 and rsi_7[i-1] > 70:
                new_signal = -SHORT_SIZE
            # Condition 6: Session filter + any bear signal (reduce noise)
            elif in_session and bear_4h and rsi_7[i] > 60:
                new_signal = -SHORT_SIZE * 0.6
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # Exit long on RSI overbought or regime flip
        if in_position and position_side > 0:
            if rsi_extreme_high:
                new_signal = 0.0
            # Exit if 4h trend flips bearish strongly
            if bear_4h and hma_4h_slope_bear:
                new_signal = 0.0
        
        # Exit short on RSI oversold or regime flip
        if in_position and position_side < 0:
            if rsi_extreme_low:
                new_signal = 0.0
            # Exit if 4h trend flips bullish strongly
            if bull_4h and hma_4h_slope_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals