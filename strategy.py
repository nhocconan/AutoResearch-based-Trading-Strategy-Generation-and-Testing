#!/usr/bin/env python3
"""
Experiment #457: 15m Multi-Regime Ensemble with 4h Trend Filter

Hypothesis: 15m timeframe needs careful regime detection to avoid noise whipsaws.
This strategy uses:

1. 4H HMA(21) TREND BIAS (via mtf_data helper):
   - Long bias when price > 4h HMA
   - Short bias when price < 4h HMA
   - HMA reduces lag vs EMA for trend detection

2. 1H ADX(14) REGIME FILTER (via mtf_data helper):
   - ADX > 25 = trending (allow breakout signals)
   - ADX < 20 = ranging (allow mean reversion only)
   - Hysteresis prevents rapid regime switching

3. 15M SIGNALS (ensemble - multiple trigger types):
   a) RSI(14) MEAN REVERSION: RSI < 30 (long) or > 70 (short)
      - Only in ranging regime + aligned with 4h trend
      - Proven edge on 15m for BTC/ETH
   
   b) MACD(12,26,9) HISTOGRAM CROSSOVER:
      - Histogram crosses above 0 = long signal
      - Histogram crosses below 0 = short signal
      - Only in trending regime + aligned with 4h trend
   
   c) BOLLINGER BAND SQUEEZE BREAKOUT:
      - BB Width < 20th percentile = squeeze
      - Price breaks upper BB = long, lower BB = short
      - High probability after compression

4. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for crash protection

5. POSITION SIZING: 0.25 discrete (conservative for 15m volatility)
   - Max 25% capital per position
   - Discrete levels minimize fee churn

Why this should work on 15m:
- Multiple signal types ensure sufficient trade frequency
- 4h HMA + 1h ADX filters prevent counter-trend disasters
- Regime-adaptive (different logic for trending vs ranging)
- Looser RSI thresholds than failed experiments
- Should work on BTC/ETH/SOL individually

Timeframe: 15m (REQUIRED for this experiment)
HTF: 1h and 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_ensemble_4h_hma_1h_adx_macd_bb_atr_v1"
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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper.values, lower.values, sma.values

def calculate_bb_width(upper, lower, sma):
    """Calculate Bollinger Band Width as percentage."""
    width = (upper - lower) / sma
    return width

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 4h HMA trend
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h ADX regime
    adx_1h = calculate_adx(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, 14)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_sma)
    
    # Calculate BB width percentile for squeeze detection
    bb_width_pct = np.full(n, np.nan)
    lookback = 100
    for i in range(lookback, n):
        valid_widths = bb_width[i-lookback:i+1]
        valid_widths = valid_widths[~np.isnan(valid_widths)]
        if len(valid_widths) > 0:
            bb_width_pct[i] = np.percentile(valid_widths, np.where(valid_widths <= bb_width[i])[0].size / len(valid_widths) * 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    # Track MACD histogram for crossover detection
    prev_macd_hist = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_1h_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(macd_hist[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 1H ADX REGIME DETECTION ===
        adx_val = adx_1h_aligned[i]
        trending_market = adx_val > 25
        ranging_market = adx_val < 20
        
        # === SIGNAL 1: RSI MEAN REVERSION (ranging market) ===
        rsi_long = rsi[i] < 30 and ranging_market
        rsi_short = rsi[i] > 70 and ranging_market
        
        # === SIGNAL 2: MACD HISTOGRAM CROSSOVER (trending market) ===
        macd_long = macd_hist[i] > 0 and prev_macd_hist <= 0 and trending_market
        macd_short = macd_hist[i] < 0 and prev_macd_hist >= 0 and trending_market
        
        # === SIGNAL 3: BB SQUEEZE BREAKOUT (any regime) ===
        squeeze = bb_width_pct[i] < 20 if not np.isnan(bb_width_pct[i]) else False
        bb_long = squeeze and close[i] > bb_upper[i-1] if i > 0 else False
        bb_short = squeeze and close[i] < bb_lower[i-1] if i > 0 else False
        
        # === GENERATE SIGNAL (Ensemble - any signal can trigger) ===
        new_signal = 0.0
        
        # RSI MEAN REVERSION (ranging market + trend aligned)
        if rsi_long and bull_trend_4h:
            new_signal = SIZE
        elif rsi_short and bear_trend_4h:
            new_signal = -SIZE
        
        # MACD CROSSOVER (trending market + trend aligned)
        if new_signal == 0.0:
            if macd_long and bull_trend_4h:
                new_signal = SIZE
            elif macd_short and bear_trend_4h:
                new_signal = -SIZE
        
        # BB SQUEEZE BREAKOUT (trend aligned)
        if new_signal == 0.0:
            if bb_long and bull_trend_4h:
                new_signal = SIZE
            elif bb_short and bear_trend_4h:
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
        
        # Store MACD histogram for next iteration crossover detection
        prev_macd_hist = macd_hist[i]
    
    return signals