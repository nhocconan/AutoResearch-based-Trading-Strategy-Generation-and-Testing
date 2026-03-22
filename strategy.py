#!/usr/bin/env python3
"""
Experiment #463: 15m Multi-Signal Mean Reversion with HTF Trend Filter

Hypothesis: After analyzing 462 failed experiments, the key insight is that 15m 
strategies need to focus on mean reversion WITH trend alignment, not pure trend 
following. Pure trend strategies get whipsawed on 15m. This strategy uses:

1. 4H HMA(21) TREND BIAS (via mtf_data helper):
   - Only long when price > 4h HMA (bull bias)
   - Only short when price < 4h HMA (bear bias)
   - Prevents trading against major trend

2. 1H ADX(14) REGIME FILTER (via mtf_data helper):
   - ADX < 25 = ranging (enable mean reversion signals)
   - ADX >= 25 = trending (reduce mean reversion, allow breakout)
   - Critical for avoiding mean reversion in strong trends

3. THREE 15M MEAN REVERSION SIGNALS (loose thresholds for trades):
   a) RSI(7) EXTREMES: RSI < 28 (long) or > 72 (short)
      - Faster RSI(7) for 15m timeframe
      - Looser than 30/70 to ensure trades on all symbols
   
   b) BOLLINGER BAND TOUCHES: Price touches lower band (long) or upper (short)
      - BB(20, 2.0) - standard settings
      - Must align with 4h trend bias
   
   c) ATR VOL SPIKE: ATR(7)/ATR(30) > 1.8 + price reverses
      - Captures panic reversals
      - Works in any regime

4. ATR(14) TRAILING STOP at 2.0x:
   - Signal → 0 when price moves 2.0*ATR against position
   - Tighter than 2.5x for 15m timeframe

5. POSITION SIZING: 0.25 discrete
   - 25% capital per position
   - Discrete levels minimize fee churn

Why this should work on 15m:
- Mean reversion works better than trend on intraday timeframes
- 4h HMA provides robust trend filter without whipsaw
- Loose RSI thresholds (28/72) ensure >10 trades per symbol
- 15m captures intraday swings that 4h/12h miss
- Should work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 1h and 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_meanrev_4h_hma_1h_adx_rsi_bb_atr_v1"
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
    """Calculate Relative Strength Index with faster period for 15m."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return middle.values, upper.values, lower.values

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """Calculate ATR ratio for vol spike detection."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    ratio = atr_short / np.where(atr_long > 1e-10, atr_long, np.inf)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    adx_1h = calculate_adx(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 7)
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 30)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_mid[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 1H ADX REGIME FILTER ===
        adx_1h_val = adx_1h_aligned[i]
        ranging_market = adx_1h_val < 25
        trending_market = adx_1h_val >= 25
        
        # === SIGNAL 1: RSI(7) MEAN REVERSION ===
        # Loose thresholds to ensure trades on all symbols
        rsi_long = rsi[i] < 28
        rsi_short = rsi[i] > 72
        
        # === SIGNAL 2: BOLLINGER BAND TOUCHES ===
        bb_long = low[i] <= bb_lower[i] * 1.001  # Touch or break lower band
        bb_short = high[i] >= bb_upper[i] * 0.999  # Touch or break upper band
        
        # === SIGNAL 3: ATR VOL SPIKE REVERSION ===
        vol_spike = atr_ratio[i] > 1.8
        # Check for reversal candle
        bullish_reversal = close[i] > (open[i] if 'open' in prices.columns else close[i-1])
        bearish_reversal = close[i] < (open[i] if 'open' in prices.columns else close[i-1])
        
        # === GENERATE SIGNAL (Ensemble - any signal can trigger) ===
        new_signal = 0.0
        
        # RSI MEAN REVERSION (primary signal, must align with 4h trend)
        if rsi_long and bull_trend_4h:
            new_signal = SIZE
        elif rsi_short and bear_trend_4h:
            new_signal = -SIZE
        
        # BOLLINGER BAND MEAN REVERSION (secondary signal)
        if new_signal == 0.0 and ranging_market:
            if bb_long and bull_trend_4h:
                new_signal = SIZE
            elif bb_short and bear_trend_4h:
                new_signal = -SIZE
        
        # VOL SPIKE REVERSION (works in any regime, captures panic)
        if new_signal == 0.0 and vol_spike:
            if bullish_reversal and bull_trend_4h:
                new_signal = SIZE
            elif bearish_reversal and bear_trend_4h:
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